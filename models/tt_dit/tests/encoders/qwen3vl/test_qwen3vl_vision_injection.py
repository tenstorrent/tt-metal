# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# How vision reaches the Qwen3-VL decoder: the tower's merged tokens REPLACE the
# <|image_pad|> row embeddings before the block stack; its deepstack features are
# ADDED to those same rows after each of the first len(deepstack) decoder layers.

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import _scatter_rows, create_rope_tensors, vision_token_runs
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.tensor import bf16_tensor
from .common import encoder_from_hf_config

IMAGE_TOKEN_ID = 151655  # <|image_pad|>
HIDDEN = 128
SEQ = 64


def test_vision_token_runs_finds_one_run_per_image():
    ids = torch.tensor([[1, 2, 3] + [IMAGE_TOKEN_ID] * 4 + [9, 9] + [IMAGE_TOKEN_ID] * 6 + [7]])
    assert vision_token_runs(ids, IMAGE_TOKEN_ID) == [(3, 4), (9, 6)]


def test_vision_token_runs_handles_the_edges():
    assert vision_token_runs(torch.tensor([[IMAGE_TOKEN_ID] * 3 + [1, 2]]), IMAGE_TOKEN_ID) == [(0, 3)]
    assert vision_token_runs(torch.tensor([[1, 2] + [IMAGE_TOKEN_ID] * 3]), IMAGE_TOKEN_ID) == [(2, 3)]
    assert vision_token_runs(torch.tensor([[1, 2, 3]]), IMAGE_TOKEN_ID) == []


def test_vision_token_runs_rejects_a_batch(expect_error):
    """One request is one sequence; a batch would silently take only the first row."""
    with expect_error(ValueError, "expected a single sequence"):
        vision_token_runs(torch.zeros(2, 8, dtype=torch.long), IMAGE_TOKEN_ID)


_MESH = [pytest.param((1, 1), (1, 1), id="single")]


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_scatter_rows_rejects_a_row_count_mismatch(mesh_device, submesh_shape, expect_error):
    """Too few or too many value rows is a caller error, not a silent partial write."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    base = bf16_tensor(torch.zeros(1, SEQ, HIDDEN), device=submesh)
    values = bf16_tensor(torch.zeros(1, 5, HIDDEN), device=submesh)
    with expect_error(ValueError, "runs cover 16 rows but values has 5"):
        _scatter_rows(base, values, [(8, 16)], add=False)


def _reference_text_model(layers):
    config = transformers.Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._from_config(config).eval()


def _encoder(submesh, *, layers, activation_layers):
    reference = _reference_text_model(layers)
    enc = encoder_from_hf_config(
        reference.config,
        head_dim=32,
        activation_layers=activation_layers,
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=0)),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
    )
    enc.load_torch_state_dict(reference.state_dict())
    return enc


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_text_only_path_is_unchanged(mesh_device, submesh_shape):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    enc = _encoder(submesh, layers=4, activation_layers=(3,))
    ids = ttnn.from_torch(
        torch.randint(0, 256, (1, SEQ)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))

    plain = enc.forward(ids, attention_mask=None, pos_embeds=pe)[0]
    explicit_none = enc.forward(
        ids, attention_mask=None, pos_embeds=pe, vision_embeds=None, vision_runs=None, deepstack_embeds=None
    )[0]
    a = tensor.to_torch(plain, mesh_axes=[None, None, None])
    b = tensor.to_torch(explicit_none, mesh_axes=[None, None, None])
    assert torch.equal(a, b), "explicitly passing None diverges from omitting the arguments"


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_vision_arguments_must_be_paired(mesh_device, submesh_shape, expect_error):
    """Embeds without runs (or the reverse) is a caller error rather than a silent no-op."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    enc = _encoder(submesh, layers=2, activation_layers=(1,))
    ids = ttnn.from_torch(
        torch.zeros(1, SEQ, dtype=torch.long), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))
    embeds = bf16_tensor(torch.zeros(1, 16, HIDDEN), device=submesh)

    with expect_error(ValueError, "must be passed together"):
        enc.forward(ids, attention_mask=None, pos_embeds=pe, vision_embeds=embeds)
    with expect_error(ValueError, "needs vision_runs"):
        enc.forward(ids, attention_mask=None, pos_embeds=pe, deepstack_embeds=[embeds])


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
def test_deepstack_is_applied_at_the_leading_layers(mesh_device, submesh_shape):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    torch.manual_seed(0)
    enc = _encoder(submesh, layers=4, activation_layers=(3,))
    ids = ttnn.from_torch(
        torch.randint(0, 256, (1, SEQ)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh
    )
    cos, sin = create_rope_tensors(1, SEQ, None, 32, 10000.0, [6, 5, 5])
    pe = (bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh))
    runs = [(8, 16)]
    vision = bf16_tensor(torch.randn(1, 16, HIDDEN), device=submesh)
    feat = [bf16_tensor(torch.randn(1, 16, HIDDEN), device=submesh) for _ in range(2)]

    def run(deepstack):
        out = enc.forward(
            ids,
            attention_mask=None,
            pos_embeds=pe,
            vision_embeds=vision,
            vision_runs=runs,
            deepstack_embeds=deepstack,
        )[0]
        return tensor.to_torch(out, mesh_axes=[None, None, None])

    none, one, two = run(None), run(feat[:1]), run(feat[:2])
    assert not torch.allclose(none, one, atol=1e-2), "a deepstack feature had no effect"
    assert not torch.allclose(one, two, atol=1e-2), "the second deepstack feature had no effect"


HIDDEN_REAL = 5120  # real conditioner width

# production keyframe runs cross 31/63 tile boundaries; runs inside one 32-row tile never exercise the slicing hazard
_TILE_RUNS = [
    pytest.param(1054, [(5, 1008)], HIDDEN_REAL, id="production_keyframe_crosses_31"),
    pytest.param(2067, [(5, 1008), (1018, 1008)], HIDDEN_REAL, id="production_two_keyframes"),
    pytest.param(SEQ, [(0, 8)], HIDDEN, id="edge_run_at_row_0"),
    pytest.param(SEQ, [(SEQ - 8, 8)], HIDDEN, id="edge_run_at_seq_end"),
]


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
@pytest.mark.parametrize(("seq", "runs", "hidden"), _TILE_RUNS)
def test_scatter_rows_is_exact_across_tile_boundaries(mesh_device, submesh_shape, seq, runs, hidden):
    """Replace is pure data movement, so torch.equal, not PCC (PCC would tolerate rows landing a tile off)."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    total = sum(length for _, length in runs)
    assert all(
        start % ttnn.TILE_SIZE or length % ttnn.TILE_SIZE for start, length in runs
    ), "every run should be unaligned on at least one end, or this gates nothing"

    torch.manual_seed(0)
    # pre-rounded through bf16 so the comparison measures the scatter, not the host's fp32 -> bf16 cast
    base = torch.randn(1, seq, hidden).bfloat16().float()
    values = torch.randn(1, total, hidden).bfloat16().float()

    golden = base.clone()
    taken = 0
    for start, length in runs:
        golden[:, start : start + length, :] = values[:, taken : taken + length, :]
        taken += length

    out = _scatter_rows(bf16_tensor(base, device=submesh), bf16_tensor(values, device=submesh), runs, add=False)
    actual = tensor.to_torch(out, mesh_axes=[None, None, None]).float()
    assert actual.shape[-2:] == (seq, hidden), f"{tuple(actual.shape)} != (…, {seq}, {hidden})"

    if not torch.equal(golden, actual):
        wrong = (golden != actual).any(dim=-1).nonzero().flatten().tolist()
        raise AssertionError(
            f"{len(wrong)} of {seq} rows differ; first 10 at {wrong[:10]} "
            f"(tile row-blocks {sorted({r // ttnn.TILE_SIZE for r in wrong[:10]})}), runs={runs}"
        )


@pytest.mark.parametrize(("mesh_device", "submesh_shape"), _MESH, indirect=["mesh_device"])
@pytest.mark.parametrize(("seq", "runs", "hidden"), _TILE_RUNS)
def test_scatter_rows_add_is_exact_across_tile_boundaries(mesh_device, submesh_shape, seq, runs, hidden):
    """add=True: untouched rows bit-exact; written rows within 2 unbiased bf16 ulps (ttnn.add rounds differently)."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    total = sum(length for _, length in runs)

    torch.manual_seed(0)
    base = torch.randn(1, seq, hidden).bfloat16().float()
    values = torch.randn(1, total, hidden).bfloat16().float()

    golden = base.clone()
    written = torch.zeros(seq, dtype=torch.bool)
    taken = 0
    for start, length in runs:
        golden[:, start : start + length, :] += values[:, taken : taken + length, :]
        written[start : start + length] = True
        taken += length

    out = _scatter_rows(bf16_tensor(base, device=submesh), bf16_tensor(values, device=submesh), runs, add=True)
    actual = tensor.to_torch(out, mesh_axes=[None, None, None]).float()

    assert torch.equal(golden[:, ~written], actual[:, ~written]), (
        f"pass-through rows changed under add; "
        f"{(golden[:, ~written] != actual[:, ~written]).any(dim=-1).sum().item()} of "
        f"{int((~written).sum())} untouched rows differ"
    )
    expected, got = golden[:, written], actual[:, written]
    # one bf16 ulp = 2**-7 of the binade; 2 ulps lets boundary entries fall either way
    ulp = torch.ldexp(torch.ones_like(expected), torch.floor(torch.log2(expected.abs().clamp(min=1e-30))).int() - 7)
    diff = (expected - got).abs()
    worst = (diff / ulp).max().item()
    assert worst <= 2.0, f"written rows are {worst:.2f} bf16 ulps out, past the 2-ulp floor"

    # No systematic bias: round-to-nearest is unbiased, truncation is not.
    bias = ((got - expected) / ulp).mean().item()
    assert abs(bias) < 0.1, f"mean error {bias:+.3f} ulps suggests truncation rather than round-to-nearest"
    logger.info(f"scatter add @ seq={seq}: worst {worst:.2f} ulps, mean bias {bias:+.3f} ulps")
