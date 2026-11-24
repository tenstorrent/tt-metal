# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace-capturable device-token embedding: exact owner selection, no host round trip."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    LOCAL_VOCAB_SIZE,
    TILE_SIZE,
    TP_SIZE,
    VOCAB_SIZE,
    Qwen38TTNNTokenEmbedding,
    Qwen38TTNNTokenRowConstants,
    _localize_token_ids_on_host,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
EMBEDDING_SOURCE = inspect.getsource(embedding_module)


def _slice(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


# --- Numerics: the device path selects the exact vocabulary owner ---------------


def _device_localized_indices(token: int) -> list[int]:
    """``embed_device_token``'s per-coordinate index formula, mirrored on the host.

    ``vocab_localize_row`` holds ``d * LOCAL_VOCAB_SIZE - 1`` at column 0, so
    coordinate d computes ``clamp(token - (d*LOCAL - 1), 0, LOCAL + 1)``.
    """

    indices = []
    for shard in range(TP_SIZE):
        shifted = token - (shard * LOCAL_VOCAB_SIZE - 1)
        indices.append(int(min(max(shifted, 0), LOCAL_VOCAB_SIZE + 1)))
    return indices


@pytest.mark.parametrize(
    "token",
    [0, 1, 17, LOCAL_VOCAB_SIZE - 1, LOCAL_VOCAB_SIZE, 62097, 124177, 186257, VOCAB_SIZE - 1],
)
def test_device_localization_matches_the_host_localizer(token: int) -> None:
    host = _localize_token_ids_on_host(torch.tensor([[token]], dtype=torch.int64))
    host_indices = host.to(torch.int64).reshape(TP_SIZE).tolist()
    assert _device_localized_indices(token) == host_indices
    # Exactly one coordinate lands on a real (nonzero-sentinel) row.
    real = [1 <= index <= LOCAL_VOCAB_SIZE for index in host_indices]
    assert sum(real) == 1
    owner = real.index(True)
    assert owner * LOCAL_VOCAB_SIZE + (host_indices[owner] - 1) == token


def test_sum_select_reproduces_the_owner_embedding_row_bitwise() -> None:
    """Summing the four sentinel lookups equals the owner's row (three are zero)."""

    torch.manual_seed(0)
    hidden = 8
    real_shards = [torch.randn(LOCAL_VOCAB_SIZE, hidden, dtype=torch.bfloat16) for _ in range(TP_SIZE)]
    full_table = torch.cat(real_shards, dim=0)
    # Sentinel-pad each shard exactly as Qwen38TTNNTokenEmbedding.sentinel_weight does.
    sentinel_shards = [
        torch.cat([torch.zeros(1, hidden, dtype=torch.bfloat16), shard, torch.zeros(1, hidden, dtype=torch.bfloat16)])
        for shard in real_shards
    ]
    for token in (0, 5, LOCAL_VOCAB_SIZE - 1, LOCAL_VOCAB_SIZE, 2 * LOCAL_VOCAB_SIZE + 9, VOCAB_SIZE - 1):
        indices = _device_localized_indices(token)
        partials = torch.stack([sentinel_shards[shard][indices[shard]] for shard in range(TP_SIZE)])
        # BF16 add of one real row and three exact zeros is bitwise the owner row.
        selected = partials.to(torch.float32).sum(dim=0).to(torch.bfloat16)
        expected = full_table[token]
        assert torch.equal(selected.view(torch.int16), expected.view(torch.int16))


# --- Structure: the embed path is token-invariant and trace-capturable ----------


def test_all_reduce_owner_select_sums_partials_then_partitions() -> None:
    helper = _slice(
        EMBEDDING_SOURCE, "def _all_reduce_owner_select_hidden(", "\ndef _all_gather_owner_select_hidden_fallback("
    )
    assert "ttnn.all_reduce(" in helper
    assert "ttnn.mesh_partition(" in helper
    assert "cluster_axis=TP_AXIS" in helper
    assert "placement=TensorPlacement.LOCAL_PARTIAL" in helper
    assert "mark_collective_shard(" in helper
    # A token-dependent slice or any host transfer would break trace identity.
    for forbidden in ("ttnn.slice(", "ttnn.all_gather(", "ttnn.to_torch(", "ttnn.from_torch(", "active_vocab_shard"):
        assert forbidden not in helper


def test_embed_device_token_is_token_invariant_owner_sum_with_no_host_io() -> None:
    body = inspect.getsource(Qwen38TTNNTokenEmbedding.embed_device_token)
    order = (
        "ttnn.subtract(token_row, constants.vocab_localize_row",
        "ttnn.clamp(shifted, min=0.0, max=float(LOCAL_VOCAB_SIZE + 1)",
        "ttnn.typecast(localized, ttnn.uint32",
        "ttnn.to_layout(localized_indices, ttnn.ROW_MAJOR_LAYOUT",
        "ttnn.reshape(row_major, (1, 1, TILE_SIZE))",
        "ttnn.embedding(",
        "self.sentinel_weight",
        "ttnn.unsqueeze_to_4D(embedded_base)",
        "mark_local_partial(",
        "_all_reduce_owner_select_hidden(",
    )
    positions = [body.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    # The token id is a device argument; nothing may be uploaded or read back.
    for forbidden in ("from_torch(", "to_torch(", "ttnn.slice("):
        assert forbidden not in body
    # Every metadata contract names what the op actually returned.
    assert body.count("_metadata(") >= 4
    # The 32-row path leaves the one-row embed alone.
    assert "vocab_localize_lanes" not in body and "rows=" not in body and "CHUNK_ROWS" not in body
    assert "ttnn.reshape(embedded_padded, ttnn.Shape((1, 1, 1, HIDDEN_SIZE)), ttnn.Shape(expected_partial))" in body


def test_upload_token_row_seeds_one_replicated_fp32_row() -> None:
    body = inspect.getsource(Qwen38TTNNTokenEmbedding.upload_token_row)
    assert "0 <= token_id < VOCAB_SIZE" in body
    assert "replicate_tensor_2d_mesh_mapper(self.mesh_device)" in body
    assert "dtype=ttnn.float32" in body
    assert "layout=ttnn.TILE_LAYOUT" in body


def test_host_embedding_and_owner_select_fallback_are_retained_for_the_eager_lanes() -> None:
    assert "def _all_gather_owner_select_hidden_fallback(" in EMBEDDING_SOURCE
    assert inspect.getsource(model_module.Qwen38TTNNTextModel._embed_residual)
    # The fallback is explicitly marked as the host-token path of the other lanes.
    fallback = _slice(
        EMBEDDING_SOURCE, "def _all_gather_owner_select_hidden_fallback(", "\ndef _localize_token_ids_on_host("
    )
    assert "Host-token path for the eager and fixed-position lanes" in fallback
    assert "never trace-captured" in fallback


# --- Structure: the runner-facing plumbing offers the device path additively ----


def test_prepared_inputs_and_forward_decode_route_the_device_token() -> None:
    prepared = inspect.getsource(model_module.Qwen38TTNNPreparedDecodeInputs)
    assert "device_token: Any | None = None" in prepared
    # release() only owns residual_sharded; the caller owns the token row.
    assert "self._owned_residual = [self.residual_sharded]" in prepared

    prepare = inspect.getsource(model_module.Qwen38TTNNTextModel.prepare_decode_inputs)
    assert "device_token=None" in prepare
    assert "if device_token is None:" in prepare
    assert 'validate_token_row(device_token, label="prepared device token row")' in prepare

    forward = inspect.getsource(model_module.Qwen38TTNNTextModel.forward_decode)
    assert "elif _prepared_inputs.device_token is not None:" in forward
    assert "self._embed_residual_from_device_token(_prepared_inputs.device_token)" in forward

    # Both residual builders use the branch-major [1,4,1,640] construction.
    embed = inspect.getsource(model_module.Qwen38TTNNTextModel._embed_residual_from_device_token)
    host_embed = inspect.getsource(model_module.Qwen38TTNNTextModel._embed_residual)
    assert "embed_device_token(token_row)" in embed
    assert "ttnn.repeat_interleave(" in embed
    assert "dim=1" in embed and "dim=2" not in embed
    assert "dim=1" in host_embed and "dim=2" not in host_embed
    assert "RESIDUAL_LOCAL_SHAPE != (1, 4, 1, 640)" in inspect.getsource(model_module)


# --- Fail-closed contracts ------------------------------------------------------


@pytest.mark.parametrize("token", [-1, VOCAB_SIZE, True, 1.0, "17"])
def test_upload_token_row_rejects_out_of_range_or_nonint_tokens(token, expect_error) -> None:
    shell = object.__new__(Qwen38TTNNTokenEmbedding)
    with expect_error(ValueError, "exact integer token"):
        shell.upload_token_row(token)


def test_validate_token_row_rejects_a_wrong_shape_row(expect_error) -> None:
    import ttnn

    shell = object.__new__(Qwen38TTNNTokenEmbedding)
    shell.mesh_contract = type("C", (), {"validate_tensor": lambda self, *a, **k: None})()

    class _Row:
        shape = (1, 1, 1, 8)
        padded_shape = (1, 1, 32, 32)
        dtype = ttnn.float32
        layout = ttnn.TILE_LAYOUT

    with expect_error(ValueError, r"FP32 TILE .* got shape=\(1, 1, 1, 8\) padded_shape=\(1, 1, 32, 32\)"):
        shell.validate_token_row(_Row())


# --- ttnn.embedding output metadata: emulate the op from its C++ rule ------------

EMBEDDING_CPP = REPO_ROOT / "ttnn/cpp/ttnn/operations/embedding/embedding.cpp"


def _ttnn_embedding_output_shape(indices_shape: tuple[int, ...], weight_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Logical output shape of ``ttnn.embedding`` as the C++ wrapper builds it.

    ``ttnn/cpp/ttnn/operations/embedding/embedding.cpp`` takes ``batch_size`` (1 for rank-1 indices, else
    ``indices[0]``) and ``sentence_size`` (``indices[-1]``) and reshapes the device op's
    ``[batch, 1, sentence, hidden]`` result to ``[sentence, hidden]`` for rank-1 indices and
    ``[batch, sentence, hidden]`` otherwise.
    """

    hidden = weight_shape[-1]
    if len(indices_shape) == 1:
        return (indices_shape[0], hidden)
    return (indices_shape[0], indices_shape[-1], hidden)


def test_embedding_output_shape_rule_matches_the_cpp_wrapper() -> None:
    source = EMBEDDING_CPP.read_text(encoding="utf-8")
    for line in (
        "auto batch_size = (input_tensor.logical_shape().rank() == 1) ? 1 : input_tensor.logical_shape()[0];",
        "auto sentence_size = input_tensor.logical_shape()[-1];",
        "embeddings = ttnn::reshape(embeddings, Shape({sentence_size, hidden_embedding_dim}));",
        "embeddings = ttnn::reshape(embeddings, Shape({batch_size, sentence_size, hidden_embedding_dim}));",
    ):
        assert line in source
    # The rank-4 device-op shape never reaches the caller.
    assert "Shape({batch_size, 1, sentence_size, hidden_embedding_dim})" not in source
    assert _ttnn_embedding_output_shape((1, 1, 32), (LOCAL_VOCAB_SIZE + 2, 2560)) == (1, 32, 2560)
    assert _ttnn_embedding_output_shape((1, 32), (LOCAL_VOCAB_SIZE + 2, 2560)) == (1, 32, 2560)
    assert _ttnn_embedding_output_shape((32,), (LOCAL_VOCAB_SIZE + 2, 2560)) == (32, 2560)


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...], *, padded_shape=None, dtype: str, layout: str) -> None:
        self.name = name
        self.shape = shape
        self.padded_shape = padded_shape or shape
        self.dtype = dtype
        self.layout = layout

    def memory_config(self) -> str:
        return "dram"


def _fake_device_token_ttnn(calls: list, deallocated: list):
    """A ttnn stand-in whose ops return the metadata the pinned runtime returns."""

    def op(name, output):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return output

        return call

    def embedding(indices, weight, **kwargs):
        calls.append(("embedding", (indices, weight), kwargs))
        assert kwargs["layout"] == "tile" and kwargs["dtype"] == "bfloat16"
        shape = _ttnn_embedding_output_shape(indices.shape, weight.shape)
        # The fused TILE program writes full 32-row tiles; 32 indices fill one tile row.
        return _FakeTensor("embedded-base", shape, dtype=weight.dtype, layout="tile")

    def unsqueeze_to_4d(value):
        calls.append(("unsqueeze_to_4D", (value,), {}))
        assert len(value.shape) < 4
        pad = (1,) * (4 - len(value.shape))
        return _FakeTensor(
            value.name + "-4d",
            pad + value.shape,
            padded_shape=pad + value.padded_shape,
            dtype=value.dtype,
            layout=value.layout,
        )

    def reshape(value, *shapes):
        calls.append(("reshape", (value, *shapes), {}))
        if len(shapes) == 1:
            return _FakeTensor(value.name + "-reshaped", tuple(shapes[0]), dtype=value.dtype, layout=value.layout)
        logical, padded = shapes
        return _FakeTensor(
            value.name + "-view", tuple(logical), padded_shape=tuple(padded), dtype=value.dtype, layout=value.layout
        )

    row = (1, 1, 1, TILE_SIZE)
    row_tile = (1, 1, TILE_SIZE, TILE_SIZE)
    return SimpleNamespace(
        float32="float32",
        uint32="uint32",
        bfloat16="bfloat16",
        ROW_MAJOR_LAYOUT="row-major",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        Topology=SimpleNamespace(Linear="linear"),
        Shape=lambda values: tuple(values),
        subtract=op("subtract", _FakeTensor("shifted", row, padded_shape=row_tile, dtype="float32", layout="tile")),
        clamp=op("clamp", _FakeTensor("localized", row, padded_shape=row_tile, dtype="float32", layout="tile")),
        typecast=op(
            "typecast", _FakeTensor("localized-indices", row, padded_shape=row_tile, dtype="uint32", layout="tile")
        ),
        to_layout=op("to_layout", _FakeTensor("row-major", row, dtype="uint32", layout="row-major")),
        reshape=reshape,
        embedding=embedding,
        unsqueeze_to_4D=unsqueeze_to_4d,
        deallocate=lambda tensor: deallocated.append(tensor.name),
    )


TOKEN_ROW = _FakeTensor("token-row", (1, 1, 1, 32), padded_shape=(1, 1, 32, 32), dtype="float32", layout="tile")


def _device_token_shell(marked: list, **token_row_constants) -> Qwen38TTNNTokenEmbedding:
    """An embedding without a device whose mesh contract records each local-partial mark in ``marked``."""

    shell = object.__new__(Qwen38TTNNTokenEmbedding)
    shell.mesh_contract = SimpleNamespace(
        validate_tensor=lambda tensor, **kwargs: None,
        mark_local_partial=lambda tensor, **kwargs: marked.append((tensor.name, kwargs["expected_shape"])),
    )
    shell.weights = SimpleNamespace(replicated_anchor="anchor", token_row=SimpleNamespace(**token_row_constants))
    shell.sentinel_weight = _FakeTensor("sentinel", (LOCAL_VOCAB_SIZE + 2, 2560), dtype="bfloat16", layout="row-major")
    shell.collective_topology = "linear"
    return shell


# The one-row chain; the 32-row chain is it minus the [1,1,1,2560] view (no second reshape, every row survives).
DEVICE_TOKEN_CALLS = [
    "subtract",
    "clamp",
    "typecast",
    "to_layout",
    "reshape",
    "embedding",
    "unsqueeze_to_4D",
    "reshape",
]


def _embed_on_fake_ttnn(monkeypatch, method: str, partial_shape, **token_row_constants):
    """``method`` of a device-less embedding on TOKEN_ROW over the fake ttnn, the owner select handed a BF16 TILE
    ``partial_shape`` partial; returns the calls (the owner select's last) and the local-partial marks."""

    calls, deallocated, marked = [], [], []
    hidden_shape = partial_shape[:3] + (640,)
    hidden = _FakeTensor("hidden", hidden_shape, padded_shape=(1, 1, 32, 640), dtype="bfloat16", layout="tile")

    def owner_select(local_partial, **kwargs):
        calls.append(("owner-select", (local_partial,), kwargs))
        assert local_partial.shape == partial_shape and local_partial.padded_shape == (1, 1, 32, 2560)
        assert local_partial.dtype == "bfloat16" and local_partial.layout == "tile"
        assert kwargs["replicated_reference"] == "anchor"
        return hidden

    monkeypatch.setattr(embedding_module, "ttnn", _fake_device_token_ttnn(calls, deallocated))
    monkeypatch.setattr(embedding_module, "_all_reduce_owner_select_hidden", owner_select)
    assert getattr(_device_token_shell(marked, **token_row_constants), method)(TOKEN_ROW) is hidden
    assert deallocated == ["shifted", "localized", "localized-indices", "row-major"]
    return calls, marked


def test_embed_device_token_consumes_the_rank_three_fused_embedding_output(monkeypatch) -> None:
    """The op returns [1, 32, 2560]; the path must still hand the collective a [1,1,1,2560] view."""

    localize = _FakeTensor("localize", (1, 1, 1, 32), dtype="float32", layout="tile")
    calls, marked = _embed_on_fake_ttnn(monkeypatch, "embed_device_token", (1, 1, 1, 2560), vocab_localize_row=localize)
    names = [call[0] for call in calls]
    assert names == DEVICE_TOKEN_CALLS + ["owner-select"] and calls[-1][2]["collective_topology"] == "linear"
    embedding_call = calls[names.index("embedding")]
    assert embedding_call[1][0].shape == (1, 1, 32) and embedding_call[1][0].dtype == "uint32"
    assert embedding_call[1][0].layout == "row-major"
    assert marked == [("embedded-base-4d-view", (1, 1, 1, 2560))]


def test_embed_device_token_names_the_actual_metadata_when_the_partial_differs(monkeypatch, expect_error) -> None:
    calls: list = []
    fake_ttnn = _fake_device_token_ttnn(calls, [])
    # A runtime whose fused embedding returned a rank-4 FP32 result would be rejected with both sides named.
    fake_ttnn.embedding = lambda indices, weight, **kwargs: _FakeTensor(
        "embedded-base", (1, 1, 32, 2560), dtype="float32", layout="tile"
    )
    monkeypatch.setattr(embedding_module, "ttnn", fake_ttnn)
    localize = _FakeTensor("localize", (1, 1, 1, 32), dtype="float32", layout="tile")
    shell = _device_token_shell([], vocab_localize_row=localize)
    with expect_error(
        RuntimeError,
        r"must be BF16 TILE \(1, 1, 32, 2560\) backed by \(1, 1, 32, 2560\), "
        r"got shape=\(1, 1, 32, 2560\) padded_shape=\(1, 1, 32, 2560\) dtype=float32 layout=tile",
    ):
        shell.embed_device_token(TOKEN_ROW)


# --- Prefill chunk: 32 lanes of one token row --------------------------------------


def test_host_token_rows_localize_per_lane_like_the_host_localizer() -> None:
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB_SIZE, (CHUNK_ROWS,)).tolist()
    ids[:4] = [0, LOCAL_VOCAB_SIZE - 1, LOCAL_VOCAB_SIZE, VOCAB_SIZE - 1]
    row = Qwen38TTNNTokenEmbedding.host_token_rows(ids)
    assert row.shape == embedding_module.TOKEN_ROW_SHAPE and row.dtype == torch.float32
    assert row.reshape(-1).to(torch.int64).tolist() == ids  # every id < 2**24 is exact in fp32
    # Coordinate d subtracts d * LOCAL_VOCAB_SIZE - 1 in every lane (vocab_localize_lanes), then clamps.
    host = _localize_token_ids_on_host(torch.tensor([ids], dtype=torch.int64)).to(torch.int64)
    for shard in range(TP_SIZE):
        device = (row.reshape(-1) - (shard * LOCAL_VOCAB_SIZE - 1)).clamp(0, LOCAL_VOCAB_SIZE + 1).to(torch.int64)
        assert torch.equal(device, host[0, shard]), shard
    for lane, token in enumerate(ids):
        assert [int(host[0, shard, lane]) for shard in range(TP_SIZE)] == _device_localized_indices(token)


@pytest.mark.parametrize(
    "ids, message",
    [
        ([0] * 31, "chunk rows must be one of"),
        ([0] * 33, "chunk rows must be one of"),
        ([0] * 31 + [VOCAB_SIZE], "token rows"),
        ([0] * 31 + [True], "token rows"),
        ([0] * 31 + [1.0], "token rows"),
    ],
)
def test_host_token_rows_reject_wrong_counts_and_out_of_range_ids(ids, message, expect_error) -> None:
    with expect_error(ValueError, message):
        Qwen38TTNNTokenEmbedding.host_token_rows(ids)


def test_token_row_constants_carry_the_lane_localizer() -> None:
    fields = tuple(Qwen38TTNNTokenRowConstants.__dataclass_fields__)
    assert fields[:2] == ("vocab_localize_row", "vocab_localize_lanes")
    build = inspect.getsource(Qwen38TTNNTokenRowConstants.build)
    assert "vocab_localize_row[..., :1].expand(1, TP_SIZE, 1, TILE_SIZE)" in build
    validate = inspect.getsource(Qwen38TTNNTokenRowConstants.validate)
    assert '("vocab_localize_row", "vocab_localize_lanes")' in validate


def test_embed_device_token_rows_keeps_all_32_rows_and_sums_per_row(monkeypatch) -> None:
    lanes = _FakeTensor("localize-lanes", (1, 1, 1, 32), dtype="float32", layout="tile")
    calls, marked = _embed_on_fake_ttnn(
        monkeypatch,
        "embed_device_token_rows",
        (1, 1, 32, 2560),
        vocab_localize_row="never-used-by-the-rows-path",
        vocab_localize_lanes=lanes,
    )
    names = [call[0] for call in calls]
    assert names == DEVICE_TOKEN_CALLS[:-1] + ["owner-select"] and calls[-1][2]["rows"] == 32
    assert calls[0][1] == (TOKEN_ROW, lanes)
    assert calls[names.index("reshape")][1][1] == (1, 1, 32)
    assert marked == [("embedded-base-4d", (1, 1, 32, 2560))]


def test_embed_device_token_rows_is_token_invariant_with_no_host_io() -> None:
    rows_body = inspect.getsource(Qwen38TTNNTokenEmbedding.embed_device_token_rows)
    for forbidden in ("from_torch(", "to_torch(", "ttnn.slice(", "ttnn.Shape("):
        assert forbidden not in rows_body
    assert "rows=rows" in rows_body and "constants.vocab_localize_lanes" in rows_body
    helper = inspect.getsource(embedding_module._all_reduce_owner_select_hidden)
    assert "rows: int = 1" in helper and "(1, 1, rows, HIDDEN_SIZE)" in helper


def test_model_embeds_residual_rows_from_the_device_token_rows() -> None:
    embed = " ".join(inspect.getsource(model_module.Qwen38TTNNTextModel._embed_residual_rows_from_device_token).split())
    assert "embed_device_token_rows(token_row)" in embed
    assert "ttnn.repeat_interleave( hidden, repeats=RESIDUAL_BRANCHES, dim=1" in embed
    assert "residual_rows_shape(rows)" in embed and "block_rows_shape(rows)" in embed
    from models.demos.blackhole.qwen38_flash_next.ttnn.layer import BLOCK_ROWS_LOCAL_SHAPE, RESIDUAL_ROWS_LOCAL_SHAPE

    assert RESIDUAL_ROWS_LOCAL_SHAPE == (1, 4, 32, 640) and BLOCK_ROWS_LOCAL_SHAPE == (1, 1, 32, 640)
    # The 1-row builder keeps the single-row embed.
    one_row = inspect.getsource(model_module.Qwen38TTNNTextModel._embed_residual_from_device_token)
    assert "embed_device_token(token_row)" in one_row and "rows" not in one_row.replace("_owners", "")
