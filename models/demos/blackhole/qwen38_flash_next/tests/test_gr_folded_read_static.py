# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the folded GR read.

The read folds the 1/4 branch mean into the RMS gamma, zero-pads the up
weight's K to the fused row width, and fuses the injection sigmoid with its x2.
These tests prove the folds against checkpoint-shaped matrices, replay the
folded read in torch next to the unfolded read (the DRAM-sharded decode read
of eec18859) at the same BF16 rounding points, and pin the device-op count per
read by walking the source.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38FinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module

QWEN_ROOT = Path(__file__).resolve().parents[1]
GR_SOURCE = QWEN_ROOT / "ttnn" / "gr.py"
FINAL_MIXER_SOURCE = QWEN_ROOT / "ttnn" / "final_mixer.py"

TP_SIZE = 4
BRANCHES = 4
HIDDEN = 2560
LOCAL_HIDDEN = HIDDEN // TP_SIZE
RANK = 320
RESIDUAL_WIDTH = BRANCHES * HIDDEN
FLAT_WIDTH = BRANCHES * LOCAL_HIDDEN
PARTIAL_WIDTH = RANK + 2 * 32
UNFOLDED_PARTIAL_WIDTH = RANK + 32  # eec18859's fused row: ten rank tiles and the injection tile
EPS = 1e-6


class _Placement:
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        residual_branches=BRANCHES,
        residual_rank=RANK,
        residual_width=RESIDUAL_WIDTH,
        rms_norm_eps=EPS,
    )

    @property
    def hidden_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((index * LOCAL_HIDDEN, (index + 1) * LOCAL_HIDDEN) for index in range(TP_SIZE))


def _random_bf16(generator: torch.Generator, *shape: int, scale: float = 0.05) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)


def _gr_source(seed: int) -> Qwen38GatedResidualWeights:
    generator = torch.Generator().manual_seed(seed)
    return Qwen38GatedResidualWeights(
        placement=_Placement(),
        layer_index=0,
        block="attn",
        norm=_random_bf16(generator, RESIDUAL_WIDTH),
        down=_random_bf16(generator, RANK, RESIDUAL_WIDTH),
        up=_random_bf16(generator, RESIDUAL_WIDTH, RANK),
        inject=_random_bf16(generator, BRANCHES, RESIDUAL_WIDTH),
    )


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16)


def _sequential_linear(row: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 ``row @ weight`` accumulating K in order, as the decode matmul does per output element."""

    accumulator = torch.zeros(row.shape[0], weight.shape[-1], dtype=torch.float32)
    for k in range(row.shape[-1]):
        accumulator = accumulator + row[:, k : k + 1].float() * weight[k : k + 1, :].float()
    return accumulator


def test_host_folds_are_exact_against_the_checkpoint_matrices() -> None:
    source = _gr_source(seed=1)
    prepared = gr_module._prepare_host_weights(source)
    norm = source.norm.unflatten(0, (BRANCHES, HIDDEN))
    generator = torch.Generator().manual_seed(11)

    # gamma / 4 is an exact exponent shift in FP32 (equals the fp64 quotient),
    # and a power-of-two scale commutes with BF16 round-to-nearest-even.
    gamma = (1.0 + norm.float()).reshape(1, BRANCHES, 1, HIDDEN)
    assert prepared["norm_scale"].dtype == torch.float32
    assert torch.equal(prepared["norm_scale"].double(), gamma.double() / BRANCHES)
    assert torch.equal(prepared["norm_scale"] * BRANCHES, gamma)
    unit = _random_bf16(generator, 1, BRANCHES, 1, HIDDEN, scale=1.0)
    assert torch.equal(_bf16(unit.float() * prepared["norm_scale"]), _bf16(unit.float() * gamma) / BRANCHES)

    # down | inject | zero pad along N (twelve tiles); up | zero pad along K to
    # the same row.  Same dtype, same bits as each device's checkpoint blocks
    # in eec18859's (device, branch, local hidden) stacking.
    down_inject = prepared["down_inject"][0, 0]
    up = prepared["up"][0, 0]
    assert down_inject.dtype == up.dtype == torch.bfloat16
    assert tuple(down_inject.shape) == (TP_SIZE * FLAT_WIDTH, PARTIAL_WIDTH)
    assert tuple(up.shape) == (PARTIAL_WIDTH, TP_SIZE * FLAT_WIDTH)
    assert torch.equal(down_inject[:, RANK + BRANCHES :], torch.zeros_like(down_inject[:, RANK + BRANCHES :]))
    assert torch.equal(up[RANK:], torch.zeros_like(up[RANK:]))
    for device in range(TP_SIZE):
        shard = source.device_shard(device)
        local = slice(device * FLAT_WIDTH, (device + 1) * FLAT_WIDTH)
        hidden = slice(device * LOCAL_HIDDEN, (device + 1) * LOCAL_HIDDEN)
        assert torch.equal(prepared["norm_scale"][0, :, 0, hidden], (1.0 + shard.norm.float()) / BRANCHES)
        assert torch.equal(down_inject[local, :RANK], shard.down.permute(1, 2, 0).reshape(FLAT_WIDTH, RANK))
        assert torch.equal(
            down_inject[local, RANK : RANK + BRANCHES], shard.inject.permute(1, 2, 0).reshape(FLAT_WIDTH, BRANCHES)
        )
        assert torch.equal(up[:RANK, local], shard.up.permute(2, 0, 1).reshape(RANK, FLAT_WIDTH))

    # fp64: the padded up matmul equals the rank-320 projection for any finite
    # values in the row's tail.
    row = _random_bf16(generator, 1, PARTIAL_WIDTH, scale=1.0).double()
    assert torch.allclose(row @ up.double(), row[:, :RANK] @ up[:RANK].double(), rtol=0, atol=1e-9)

    # The final mixer takes the same gamma fold on its own weights.
    mixer = Qwen38FinalMixerWeights(placement=_Placement(), norm=source.norm, down=source.down, up=source.up)
    mixer_prepared = final_mixer_module._prepare(mixer)
    assert torch.equal(mixer_prepared["norm_scale"] * BRANCHES, gamma)
    assert torch.equal(mixer_prepared["down"][0, 0], down_inject[:, :RANK])
    assert torch.equal(mixer_prepared["up"][0, 0], up[:RANK])


def test_folded_read_is_bit_identical_to_the_unfolded_decode_read() -> None:
    source = _gr_source(seed=2)
    prepared = gr_module._prepare_host_weights(source)
    reference = Qwen38GatedResidual(source)
    generator = torch.Generator().manual_seed(3)
    residual = _random_bf16(generator, 1, RESIDUAL_WIDTH, scale=1.0)
    shards = reference.shard_residual(residual)
    reference_blocks, reference_state = reference.read_tp4(shards)
    locals_ = [shard.unflatten(-1, (BRANCHES, LOCAL_HIDDEN)) for shard in shards]  # [1, 4, 640] per device

    # Distributed RMS as the device computes it: BF16 unit, then one BF16
    # rounding after the FP32 gamma multiply.  The folded gamma yields exactly
    # the unfolded normalized residual divided by four; both read as the flat
    # (branch, local hidden) row.
    variance = sum(local.float().square().sum(dim=-1, keepdim=True) for local in locals_) / HIDDEN
    inverse_rms = torch.rsqrt(variance + EPS)
    flat = []
    flat_q = []
    for device, local in enumerate(locals_):
        hidden = slice(device * LOCAL_HIDDEN, (device + 1) * LOCAL_HIDDEN)
        gamma = 1.0 + source.device_shard(device).norm.float()
        unit = _bf16(local.float() * inverse_rms)
        flat.append(_bf16(unit.float() * gamma).flatten(-2))
        flat_q.append(_bf16(unit.float() * prepared["norm_scale"][0, :, 0, hidden]).flatten(-2))
        assert torch.equal(flat_q[device], flat[device] / BRANCHES)

    # Unfolded (eec18859): one K=2560 FP32 chain per device over the 352-wide
    # stacked weight, summed over TP4, one BF16 cast, exact x1/4, two slices.
    # Folded: the same chain over the 384-wide weight on the quarter-scaled
    # row, the same sum, one BF16 cast, no scale, one slice.  The extra zero
    # weight columns and the zero K rows contribute exact zeros.
    down_inject = prepared["down_inject"][0, 0]
    up_pad = prepared["up"][0, 0]
    partial_sum = torch.zeros(1, UNFOLDED_PARTIAL_WIDTH)
    partial_sum_q = torch.zeros(1, PARTIAL_WIDTH)
    for device in range(TP_SIZE):
        local = slice(device * FLAT_WIDTH, (device + 1) * FLAT_WIDTH)
        partial_sum = partial_sum + _sequential_linear(flat[device], down_inject[local, :UNFOLDED_PARTIAL_WIDTH])
        partial_sum_q = partial_sum_q + _sequential_linear(flat_q[device], down_inject[local])
    scaled = _bf16(partial_sum) * (1.0 / BRANCHES)
    scaled_down = scaled[:, :RANK]
    inject_scaled = scaled[:, RANK : RANK + BRANCHES]
    row = _bf16(partial_sum_q)
    assert torch.equal(row[:, :UNFOLDED_PARTIAL_WIDTH], scaled)
    assert torch.equal(
        row[:, UNFOLDED_PARTIAL_WIDTH:], torch.zeros(1, PARTIAL_WIDTH - UNFOLDED_PARTIAL_WIDTH).to(torch.bfloat16)
    )
    inject_row = row[:, RANK : RANK + BRANCHES]
    assert torch.equal(inject_row, inject_scaled)

    # SiLU on the sliced rank row versus the whole row; the padded up matmul
    # adds exact zeros after K=320.
    low_rank = _bf16(F.silu(scaled_down.float()))
    low_rank_q = _bf16(F.silu(row.float()))
    assert torch.equal(low_rank_q[:, :RANK], low_rank)
    injection = _bf16(torch.sigmoid(inject_scaled.float())) * 2
    injection_q = _bf16(torch.sigmoid(inject_row.float())) * 2
    assert torch.equal(injection_q, injection)
    for device in range(TP_SIZE):
        local = slice(device * FLAT_WIDTH, (device + 1) * FLAT_WIDTH)
        up = _bf16(_sequential_linear(low_rank, up_pad[:RANK, local]))
        up_q = _bf16(_sequential_linear(low_rank_q, up_pad[:, local]))
        assert torch.equal(up_q, up)
        gate = _bf16(torch.sigmoid(up.float())).unflatten(-1, (BRANCHES, LOCAL_HIDDEN))
        gated = _bf16(gate.float() * flat[device].unflatten(-1, (BRANCHES, LOCAL_HIDDEN)).float())
        gated_q = _bf16(gate.float() * flat_q[device].unflatten(-1, (BRANCHES, LOCAL_HIDDEN)).float())
        assert torch.equal(gated_q, gated / BRANCHES)
        block_sum = torch.zeros(1, LOCAL_HIDDEN)
        block_sum_q = torch.zeros(1, LOCAL_HIDDEN)
        for branch in range(BRANCHES):
            block_sum = block_sum + gated[:, branch].float()
            block_sum_q = block_sum_q + gated_q[:, branch].float()
        block = _bf16(block_sum) * (1.0 / BRANCHES)
        block_q = _bf16(block_sum_q)
        assert torch.equal(block_q, block)
        # Sanity against the CPU oracle (BF16 per-device partials, so tolerance).
        torch.testing.assert_close(block_q.float(), reference_blocks[device].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(injection_q.float(), reference_state.injection.float(), atol=2e-2, rtol=2e-2)


def _tile_pages(padded: torch.Tensor) -> torch.Tensor:
    """Tiles of a padded 4-D tensor in TILE-layout page order, each flattened."""

    batch, channels, rows, columns = padded.shape
    tiles = padded.reshape(batch, channels, rows // 32, 32, columns // 32, 32).permute(0, 1, 2, 4, 3, 5)
    return tiles.reshape(-1, 32 * 32)


def test_flat_row_producers_write_the_matmul_shards_over_the_same_tile_pages() -> None:
    """Why the gamma multiply and the gate multiply run on the flat row and stay bit-identical.

    The branch-major residual [1,4,1,640] and its flat view [1,1,1,2560] hold
    the same 80 tile pages in the same order, so an elementwise op reads and
    writes the same (page, position) pairs through either view: the gamma
    multiply and the gate multiply on the flat row consume and produce the
    same values as on the branch-major tensor, and only the output buffer's
    page placement (five L1 shards instead of interleaved DRAM) changes.  The
    down+inject activation shard is five 512-column shards of the flat row,
    which a 640-column branch-major view cannot express, hence the flat row.
    """

    generator = torch.Generator().manual_seed(5)
    padded_branch_major = torch.zeros(1, BRANCHES, 32, LOCAL_HIDDEN, dtype=torch.bfloat16)
    padded_branch_major[:, :, 0, :] = _random_bf16(generator, 1, BRANCHES, LOCAL_HIDDEN, scale=1.0)
    # The flat view is a metadata rewrite: page p of [1,4,32,640] is page p of
    # [1,1,32,2560] ((branch b, tile j) -> 20 b + j), and element (b, k) sits
    # at the same position inside that page.  (A torch reshape of the padded
    # tensor would interleave the pad rows; build the padded flat row itself.)
    padded_flat = torch.zeros(1, 1, 32, FLAT_WIDTH, dtype=torch.bfloat16)
    padded_flat[:, :, 0, :] = padded_branch_major[:, :, 0, :].reshape(1, 1, 1, FLAT_WIDTH)
    assert torch.equal(_tile_pages(padded_branch_major), _tile_pages(padded_flat))
    assert _tile_pages(padded_flat).shape[0] == 80

    # Elementwise products through either view are the same bits.
    gamma = torch.randn(1, BRANCHES, 1, LOCAL_HIDDEN, generator=generator)
    gate = _random_bf16(generator, 1, BRANCHES, 1, LOCAL_HIDDEN, scale=1.0)
    unit = padded_branch_major[:, :, :1, :]
    normalized = _bf16(unit.float() * gamma)
    normalized_flat = _bf16(unit.reshape(1, 1, 1, FLAT_WIDTH).float() * gamma.reshape(1, 1, 1, FLAT_WIDTH))
    assert torch.equal(normalized_flat.view(torch.int16), normalized.reshape(1, 1, 1, FLAT_WIDTH).view(torch.int16))
    gated = _bf16(gate.float() * normalized.float())
    gated_flat = _bf16(normalized_flat.float() * gate.reshape(1, 1, 1, FLAT_WIDTH).float())
    assert torch.equal(gated_flat.reshape(1, BRANCHES, 1, LOCAL_HIDDEN).view(torch.int16), gated.view(torch.int16))

    # The shard the producers write is the matmul activation config itself:
    # the down+inject shard divides the padded flat row (binary_ng and unary
    # treat a shard as uneven when the padded rows or the width are not
    # multiples of the shard), and does not divide the branch-major width; the
    # up shard divides the padded fused row.
    import ttnn

    mesh_device = SimpleNamespace(dram_grid_size=lambda: ttnn.CoreCoord(8, 1))
    down_inject_act, _ = gr_module.dram_sharded_matmul_configs(mesh_device, FLAT_WIDTH, PARTIAL_WIDTH, num_cores=5)
    up_act, _ = gr_module.dram_sharded_matmul_configs(mesh_device, PARTIAL_WIDTH, FLAT_WIDTH, num_cores=2)
    for config in (down_inject_act, up_act):
        assert config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        assert config.buffer_type == ttnn.BufferType.L1
        assert config.shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    assert tuple(down_inject_act.shard_spec.shape) == (32, 512) and down_inject_act.shard_spec.grid.num_cores() == 5
    assert tuple(up_act.shard_spec.shape) == (32, 192) and up_act.shard_spec.grid.num_cores() == 2
    assert FLAT_WIDTH % 512 == 0 and PARTIAL_WIDTH % 192 == 0
    assert LOCAL_HIDDEN % 512 != 0
    mixer_up_act, _ = gr_module.dram_sharded_matmul_configs(mesh_device, RANK, FLAT_WIDTH, num_cores=2)
    assert tuple(mixer_up_act.shard_spec.shape) == (32, 160) and RANK % 160 == 0


def _attribute_name(node: ast.AST) -> str:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def _methods(source: Path, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    assert len(classes) == 1
    return {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}


# Device programs launched per ttnn call, from the lean-v2 per-op profile:
# rms_norm_pre_all_gather inserts a FillPad before its kernel; every reshape in
# the read keeps the padded shape and ttnn.experimental.view rewrites the
# tensor spec over the same buffer (metadata only); ttnn.all_reduce on FP32 is
# AllBroadcast + FastReduceNC.  Every other ttnn call is one program.  (The
# write's reshape moves four values into four tiles and is a kernel; the write
# stays at three device ops.)
DEVICE_OPS_PER_CALL = {
    "ttnn.rms_norm_pre_all_gather": 2,
    "ttnn.reshape": 0,
    "ttnn.experimental.view": 0,
    "ttnn.all_reduce": 2,
}
NOT_OPS = {"ttnn.UnaryWithParam"}
MATMULS = {"ttnn.linear"}
COLLECTIVES = {"ttnn.all_gather", "ttnn.experimental.all_gather_async"}


def _walk_ops(methods: dict[str, ast.FunctionDef], name: str) -> list[str]:
    """ttnn calls of one method in source order, with ``self.<helper>(...)`` calls inlined."""

    calls = sorted(
        (node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)),
        key=lambda call: (call.lineno, call.col_offset),
    )
    ops: list[str] = []
    for call in calls:
        callee = _attribute_name(call.func)
        if callee.startswith("self.") and callee[len("self.") :] in methods:
            ops.extend(_walk_ops(methods, callee[len("self.") :]))
        elif callee.startswith("ttnn.") and callee not in NOT_OPS:
            ops.append(callee)
    return ops


def _device_ops(ops: list[str]) -> int:
    return sum(DEVICE_OPS_PER_CALL.get(op, 1) for op in ops)


def test_read_is_eighteen_device_ops_with_two_matmuls_and_two_collectives() -> None:
    # eec18859 walked the same way: 24 device ops (its two x1/4 multiplies,
    # its rank slice and its separate injection sigmoid are the four removed
    # by the folds); 322b503a: 20 (the two interleaved-to-sharded copies in
    # front of the matmuls are the two removed here).  The walker orders calls
    # by position, so the gamma multiply precedes the flat view it takes as
    # its first argument.
    methods = _methods(GR_SOURCE, "Qwen38TTNNGatedResidual")
    ops = _walk_ops(methods, "read")
    assert ops == [
        # _normalize: the gamma multiply writes the down+inject activation shard
        "ttnn.rms_norm_pre_all_gather",
        "ttnn.reshape",
        "ttnn.all_gather",
        "ttnn.rms_norm_post_all_gather",
        "ttnn.multiply",
        "ttnn.experimental.view",
        # fused down|inject decode matmul
        "ttnn.linear",
        "ttnn.to_memory_config",
        # _all_reduce_partial
        "ttnn.experimental.all_gather_async",
        "ttnn.experimental.fast_reduce_nc",
        "ttnn.reshape",
        # row epilogue: the SiLU writes the up activation shard
        "ttnn.typecast",
        "ttnn.slice",
        "ttnn.silu",
        # up decode matmul
        "ttnn.linear",
        "ttnn.to_memory_config",
        # gate (flat, reading the normalized shard), branch mean, injection
        "ttnn.sigmoid",
        "ttnn.multiply",
        "ttnn.experimental.view",
        "ttnn.experimental.fast_reduce_nc",
        "ttnn.reshape",
        "ttnn.multiply",
    ]
    assert _device_ops(ops) == 18
    assert "ttnn.to_memory_config" not in ops[: ops.index("ttnn.linear")]
    assert ops.count("ttnn.to_memory_config") == 2
    assert sum(op in MATMULS for op in ops) == 2
    assert sum(op in COLLECTIVES for op in ops) == 2
    assert not {"ttnn.sum", "ttnn.concat", "ttnn.permute", "ttnn.repeat"} & set(ops)
    assert _walk_ops(methods, "write") == ["ttnn.reshape", "ttnn.multiply", "ttnn.add"]


def test_final_mixer_is_sixteen_device_ops_with_the_same_folds() -> None:
    # eec18859 walked the same way: 20 device ops (its two x1/4 multiplies
    # removed); 322b503a: 18 (the two interleaved-to-sharded copies removed here).
    methods = _methods(FINAL_MIXER_SOURCE, "Qwen38TTNNFinalMixer")
    ops = _walk_ops(methods, "__call__")
    assert ops == [
        "ttnn.rms_norm_pre_all_gather",
        "ttnn.reshape",
        "ttnn.all_gather",
        "ttnn.rms_norm_post_all_gather",
        "ttnn.multiply",
        "ttnn.experimental.view",
        "ttnn.linear",
        "ttnn.to_memory_config",
        "ttnn.all_reduce",
        "ttnn.typecast",
        "ttnn.silu",
        "ttnn.linear",
        "ttnn.to_memory_config",
        "ttnn.sigmoid",
        "ttnn.multiply",
        "ttnn.experimental.view",
        "ttnn.experimental.fast_reduce_nc",
        "ttnn.reshape",
    ]
    assert _device_ops(ops) == 16
    assert ops.count("ttnn.to_memory_config") == 2
    assert "ttnn.sum" not in ops


def test_producers_write_the_matmul_activation_layouts_directly() -> None:
    """The gamma multiply and the SiLU take the decode matmul activation configs as their output."""

    for source, class_name, method, down_config in (
        (GR_SOURCE, "Qwen38TTNNGatedResidual", "read", "self.down_inject_act_memory_config"),
        (FINAL_MIXER_SOURCE, "Qwen38TTNNFinalMixer", "__call__", "self.down_act_memory_config"),
    ):
        methods = _methods(source, class_name)
        calls = [
            call for name in ("_normalize", method) for call in ast.walk(methods[name]) if isinstance(call, ast.Call)
        ]
        by_name: dict[str, list[ast.Call]] = {}
        for call in calls:
            by_name.setdefault(_attribute_name(call.func), []).append(call)
        (gamma,) = [call for call in by_name["ttnn.multiply"] if ast.unparse(call.args[1]) == "self.norm_scale_flat"]
        assert ast.unparse(gamma.args[0]) == "ttnn.experimental.view(unit, FLAT_LOCAL_SHAPE)"
        assert {keyword.arg: ast.unparse(keyword.value) for keyword in gamma.keywords} == {
            "dtype": "ttnn.bfloat16",
            "memory_config": down_config,
        }
        (silu,) = by_name["ttnn.silu"]
        assert {keyword.arg: ast.unparse(keyword.value) for keyword in silu.keywords} == {
            "memory_config": "self.up_act_memory_config"
        }
        # The only remaining copies move the two matmul outputs back to DRAM.
        assert [ast.unparse(call.args[1]) for call in by_name["ttnn.to_memory_config"]] == [
            "ttnn.DRAM_MEMORY_CONFIG",
            "ttnn.DRAM_MEMORY_CONFIG",
        ]
        text = source.read_text(encoding="utf-8")
        assert "self.norm_scale_flat = ttnn.experimental.view(weights.norm_scale, FLAT_LOCAL_SHAPE)" in text
        assert "to_memory_config(normalized" not in text and "to_memory_config(low_rank" not in text


def test_fused_injection_sigmoid_uses_the_standalone_sigmoid_parameters() -> None:
    import ttnn

    methods = _methods(GR_SOURCE, "Qwen38TTNNGatedResidual")
    read_source = ast.get_source_segment(GR_SOURCE.read_text(encoding="utf-8"), methods["read"])
    assert read_source is not None
    fused = [
        call
        for call in ast.walk(methods["read"])
        if isinstance(call, ast.Call)
        and _attribute_name(call.func) == "ttnn.multiply"
        and any(keyword.arg == "input_tensor_a_activations" for keyword in call.keywords)
    ]
    assert [[ast.unparse(argument) for argument in call.args] for call in fused] == [["inject_row", "2.0"]]
    keywords = {keyword.arg: keyword.value for keyword in fused[0].keywords}
    assert set(keywords) == {"input_tensor_a_activations", "memory_config"}
    assert ast.unparse(keywords["input_tensor_a_activations"]) == (
        "[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)]"
    )
    assert "ttnn.sigmoid(inject" not in read_source

    # The pinned runtime lowers both to sigmoid_tile<VectorMode::RC, 0u>:
    # ttnn.sigmoid defaults to vector_mode=4 and SigmoidMode.Accurate, and
    # every ttnn.multiply overload (tensor and scalar rhs) takes the fused
    # input activation list.
    param = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)
    assert param.op_type == ttnn.UnaryOpType.SIGMOID
    assert "params=[4, 0]" in repr(param)
    sigmoid_doc = ttnn.sigmoid.__doc__ or ""
    assert "vector_mode (int, optional)" in sigmoid_doc and "Defaults to 4" in sigmoid_doc
    assert "Defaults to `SigmoidMode.Accurate`" in sigmoid_doc
    signatures = [entry[0] for entry in type(ttnn.multiply.function).__call__.__nb_signature__]
    assert len(signatures) >= 2
    assert all("input_tensor_b: int | int | float" in signature for signature in signatures[:1])
    assert all(
        "input_tensor_a_activations: collections.abc.Sequence[ttnn._ttnn.activation.EltwiseUnaryWithParam]" in signature
        for signature in signatures
    )
