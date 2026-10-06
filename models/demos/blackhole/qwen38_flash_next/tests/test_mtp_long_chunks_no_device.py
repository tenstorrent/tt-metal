# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts of the MTP chain's 128-row prefill chunks (``--long-chunks`` with ``--mtp``): the input
mixer's rows form at 128 rows, the MTP chunk extension's 128-row twin, the driver's mixed tail, the chain's lifted
refusals and the admission's long-chunk terms."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step4_rows_no_device import (
    BF16,
    TILE,
    TP,
    FakeTensor,
    _bf16,
    _cat,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step5_verify_no_device import (  # noqa: F401
    _mtp_input,
    fake,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROW_COUNTS, CHUNK_ROWS, LONG_CHUNK_ROWS


def _shard(host: torch.Tensor) -> FakeTensor:
    return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=3)], BF16, TILE, 3)


def _rows(tensor: FakeTensor, start: int, stop: int) -> torch.Tensor:
    return _cat(tensor, 3)[:, :, start:stop].contiguous()


# --------------------------------------------------------------------------- the input mixer at 128 rows


def test_mtp_input_mixer_128_rows_are_bitwise_the_32_row_form_per_tile_and_the_one_row_mixer_per_row(fake) -> None:
    """The 128-row form's row j is the one-row mixer's result for row j and the 32-row form's row j % 32 on its tile:
    the norms are per row and the projections run per 32-row tile."""

    mixer = _mtp_input(fake)
    torch.manual_seed(95)
    embedding, residual = _bf16(1, 1, LONG_CHUNK_ROWS, 2560), _bf16(1, 4, LONG_CHUNK_ROWS, 2560)
    rows = mixer.rows(_shard(embedding), _shard(residual))
    assert rows.shape == (1, 4, LONG_CHUNK_ROWS, 640) and rows.dtype is BF16
    for tile in range(LONG_CHUNK_ROWS // CHUNK_ROWS):
        start, stop = tile * CHUNK_ROWS, (tile + 1) * CHUNK_ROWS
        short = mixer.rows(_shard(embedding[:, :, start:stop]), _shard(residual[:, :, start:stop]))
        assert short.shape == (1, 4, CHUNK_ROWS, 640)
        assert torch.equal(_rows(rows, start, stop).view(torch.int16), _cat(short, 3).view(torch.int16)), tile
    for row in (0, 1, 31, 32, 63, 64, 95, 96, 127):
        one = mixer(_shard(embedding[:, :, row : row + 1]), _shard(residual[:, :, row : row + 1]))
        assert torch.equal(_rows(rows, row, row + 1).view(torch.int16), _cat(one, 3).view(torch.int16)), row


def test_mtp_input_mixer_rows_admit_the_chunk_row_counts_only(expect_error, fake) -> None:
    mixer = _mtp_input(fake)
    assert CHUNK_ROW_COUNTS == (32, 128)
    with expect_error(ValueError, match="token rows"):  # allow-pytest.raises: pure contract test
        mixer.rows(_shard(_bf16(1, 1, 64, 2560)), _shard(_bf16(1, 4, 64, 2560)))


def test_mtp_input_mixer_projects_per_32_row_tile_above_one_tile() -> None:
    """The 32-row call is unchanged (one ``ttnn.linear`` per projection); 128 rows slice the gathered rows per tile,
    project each tile with the same call and concatenate."""

    project = inspect.getsource(mtp_module.Qwen38TTNNMTPInput._project_rows)
    assert "if rows == ttnn.TILE_SIZE:\n            return project(gathered)" in project
    assert "for tile in range(rows // ttnn.TILE_SIZE):" in project
    assert "ttnn.slice(" in project and "ttnn.concat(projected, dim=2, memory_config=dram)" in project
    rows = inspect.getsource(mtp_module.Qwen38TTNNMTPInput.rows)
    assert "rows = _shape(input_embedding_rows)[2]" in rows and "if rows not in CHUNK_ROW_COUNTS:" in rows
    assert rows.count("self._project_rows(") == 2 and "ttnn.linear(" not in rows


# --------------------------------------------------------------------------- the chunk extension's 128-row twin


class _FakeLayer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def allocate_chunk_state(self, constants, *, base=None, local_combine_output=None):
        self.calls.append(("allocate", constants.rows, base, local_combine_output))
        return SimpleNamespace(rows=constants.rows, base=base)

    def release_chunk_state(self, state) -> None:
        self.calls.append(("release", state.rows))

    def forward_chunk_generic(self, mixed, generic_state, chunk_state, **kwargs):
        self.calls.append(("forward", mixed, chunk_state.rows, kwargs))
        return SimpleNamespace(name="out")


class _FakeEmbedding:
    def __init__(self) -> None:
        self.hosts: list[torch.Tensor] = []

    def upload_token_rows(self, rows: int):
        return SimpleNamespace(shape=(1, 1, rows // CHUNK_ROWS, CHUNK_ROWS), rows=rows)

    @staticmethod
    def host_token_rows(token_ids):
        return embedding_module.Qwen38TTNNTokenEmbedding.host_token_rows(token_ids)

    def embed_device_token_rows(self, token_row):
        return SimpleNamespace(shape=(1, 1, token_row.rows, 640), name="embedding")


def _alignment(layer: _FakeLayer):
    mixer = SimpleNamespace(calls=[])
    mixer.rows = lambda embedding, residual: mixer.calls.append((embedding.shape, residual.shape)) or SimpleNamespace(
        shape=residual.shape, name="mixed"
    )
    return SimpleNamespace(layer=layer, input_mixer=mixer, generic_state="generic")


def _extension_fixtures(monkeypatch):
    layer = _FakeLayer()
    alignment = _alignment(layer)
    embedding = _FakeEmbedding()
    copies: list = []

    def allocate_feature_rows(rows: int):
        return SimpleNamespace(tensor=SimpleNamespace(shape=(1, 1, rows, 640), name="features"), rows=rows, clean=True)

    def write_feature_rows(feature_rows, host_rows) -> None:
        # the model's rule: an image's rows land, a text chunk after an image restores the clean rows, else nothing
        if host_rows is not None:
            copies.append((host_rows, feature_rows.tensor))
            feature_rows.clean = False
        elif not feature_rows.clean:
            copies.append(("clean", feature_rows.tensor))
            feature_rows.clean = True

    model = SimpleNamespace(
        model_io=SimpleNamespace(embedding=embedding),
        mesh_device="mesh",
        _allocate_chunk_feature_rows=allocate_feature_rows,
        host_chunk_feature_rows=lambda image: image,
        write_chunk_feature_rows=write_feature_rows,
    )
    verify = SimpleNamespace(alignment=alignment)
    monkeypatch.setattr(mtp_v2, "_validate_verify_state", lambda model, verify: None)
    monkeypatch.setattr(mtp_v2, "_shape", lambda tensor: tuple(tensor.shape))
    monkeypatch.setattr(mtp_v2, "tensor_metadata", lambda tensor: tuple(tensor.shape))
    monkeypatch.setattr(mtp_v2, "_deallocate", lambda *tensors: None)
    fake_ttnn = SimpleNamespace(
        from_torch=lambda host, **kwargs: host,
        copy_host_to_device_tensor=lambda host, target: copies.append((host, target)),
        add=lambda rows, features, **kwargs: SimpleNamespace(shape=tuple(rows.shape), name="spliced"),
        float32="float32",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        deallocate=lambda *tensors: None,
    )
    monkeypatch.setattr(mtp_v2, "ttnn", fake_ttnn)
    monkeypatch.setattr(mtp_v2, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate")
    return layer, alignment, model, verify, copies


def _chunk_states():
    short = SimpleNamespace(rows=CHUNK_ROWS, rows_constants=SimpleNamespace(rows=CHUNK_ROWS), local_combine_output=None)
    long = SimpleNamespace(
        rows=LONG_CHUNK_ROWS, rows_constants=SimpleNamespace(rows=LONG_CHUNK_ROWS), local_combine_output="combine128"
    )
    return short, long


def test_extension_allocate_takes_the_chunk_state_form_and_the_128_row_twin_needs_the_32_row_base(
    expect_error, monkeypatch
) -> None:
    layer, alignment, model, verify, _ = _extension_fixtures(monkeypatch)
    short, long = _chunk_states()
    ext32 = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, short)
    assert ext32.rows == CHUNK_ROWS and ext32.token_row.shape == (1, 1, 1, 32) and ext32.alignment is alignment
    ext128 = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long, base=ext32)
    assert ext128.rows == LONG_CHUNK_ROWS and ext128.token_row.shape == (1, 1, 4, 32)
    assert layer.calls == [
        ("allocate", CHUNK_ROWS, None, None),
        ("allocate", LONG_CHUNK_ROWS, ext32.layer_chunk_state, "combine128"),
    ]
    for chunk_state, base in ((long, None), (short, ext32), (SimpleNamespace(rows=64, local_combine_output="x"), None)):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk_state, base=base)
    # the base must be the 32-row twin of the same alignment
    other = SimpleNamespace(rows=CHUNK_ROWS, alignment=object(), layer_chunk_state=None)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long, base=other)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long, base=ext128)
    # a 128-row chunk state without the shared combine buffer, a 32-row one with it: refused
    for chunk_state in (
        SimpleNamespace(rows=LONG_CHUNK_ROWS, rows_constants=long.rows_constants, local_combine_output=None),
        SimpleNamespace(rows=CHUNK_ROWS, rows_constants=short.rows_constants, local_combine_output="combine"),
    ):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(
                model, verify, chunk_state, base=ext32 if chunk_state.rows != CHUNK_ROWS else None
            )


def test_extension_128_row_twin_writes_128_tokens_runs_128_row_roots_and_refuses_the_handoff(
    expect_error, monkeypatch
) -> None:
    layer, alignment, model, verify, copies = _extension_fixtures(monkeypatch)
    short, long = _chunk_states()
    ext32 = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, short)
    ext128 = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, long, base=ext32)
    tokens = [1000 + index for index in range(LONG_CHUNK_ROWS)]
    ext128.write_tokens(model, tokens)
    host, target = copies[-1]
    assert target is ext128.token_row and host.shape == (1, 1, 4, 32) and host.reshape(-1).tolist() == tokens
    ext32.write_tokens(model, tokens[:CHUNK_ROWS])
    assert copies[-1][0].shape == (1, 1, 1, 32)
    # an image prompt's tokens ahead: the pads embed the zero sentinel and the feature rows carry the tower's rows
    # there (the backbone chunk's form one position ahead); the next text chunk restores the clean rows once
    from models.demos.blackhole.qwen38_flash_next.mrope import IMAGE_TOKEN_ID
    from models.demos.blackhole.qwen38_flash_next.vision_splice import (
        IMAGE_LANE_SENTINEL_TOKEN,
        NEGATIVE_ZERO_BF16_BITS,
    )

    image_tokens = tokens[:CHUNK_ROWS]
    image_tokens[5:9] = [IMAGE_TOKEN_ID] * 4
    features = torch.arange(4 * 2560, dtype=torch.float32).reshape(4, 2560).to(torch.bfloat16)
    ext32.write_tokens(model, image_tokens, features=features)
    (token_host, token_target), (feature_host, feature_target) = copies[-2:]
    assert token_target is ext32.token_row and token_host.reshape(-1)[5:9].tolist() == [IMAGE_LANE_SENTINEL_TOKEN] * 4
    assert token_host.reshape(-1)[:5].tolist() == image_tokens[:5]
    assert feature_target is ext32.feature_rows.tensor and feature_host.shape == (1, 1, CHUNK_ROWS, 2560)
    assert torch.equal(feature_host[0, 0, 5:9], features) and ext32.feature_rows.clean is False
    assert feature_host[0, 0, 0].view(torch.int16).tolist() == [NEGATIVE_ZERO_BF16_BITS] * 2560
    ext32.write_tokens(model, tokens[:CHUNK_ROWS])
    assert copies[-1] == ("clean", ext32.feature_rows.tensor) and ext32.feature_rows.clean is True
    cleans = len(copies)
    ext32.write_tokens(model, tokens[:CHUNK_ROWS])
    # a text chunk after a text chunk copies its token row and no feature rows
    assert len(copies) == cleans + 1 and copies[-1][1] is ext32.token_row
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        ext32.write_tokens(model, image_tokens, features=features[:3])
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        ext128.write_tokens(model, tokens[:CHUNK_ROWS])
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        ext32.write_tokens(model, tokens)
    roots = SimpleNamespace(shape=(1, 4, LONG_CHUNK_ROWS, 640))
    ext128.forward_chunk_rows(
        model, roots, rope_rows="rope", qsa_chunk="qsa", qsa_chunk_constants="constants128", selectors=None
    )
    assert alignment.input_mixer.calls == [((1, 1, LONG_CHUNK_ROWS, 640), (1, 4, LONG_CHUNK_ROWS, 640))]
    forward = layer.calls[-1]
    assert forward[0] == "forward" and forward[1].name == "mixed" and forward[2] == LONG_CHUNK_ROWS
    assert forward[3] == {
        "prepared_ple_rows": None,
        "rope_rows": "rope",
        "qsa_chunk": "qsa",
        "qsa_chunk_constants": "constants128",
        "selectors": None,
    }
    with expect_error(RuntimeError):  # allow-pytest.raises: pure contract test
        ext128.forward_chunk_rows(
            model,
            SimpleNamespace(shape=(1, 4, CHUNK_ROWS, 640)),
            rope_rows="rope",
            qsa_chunk="qsa",
            qsa_chunk_constants="constants128",
            selectors=None,
        )
    with expect_error(RuntimeError):  # allow-pytest.raises: pure contract test
        ext32.forward_chunk_rows(
            model, roots, rope_rows="rope", qsa_chunk="qsa", qsa_chunk_constants="c", selectors="s"
        )
    with expect_error(ValueError, match="hand-off"):  # allow-pytest.raises: pure contract test
        ext128.finish_chunk(model, prefilled=LONG_CHUNK_ROWS)


def test_model_chunk_body_takes_the_extension_of_the_chunk_state_form() -> None:
    body = inspect.getsource(model_module.Qwen38TTNNTextModel.forward_prefill_chunk_generic)
    assert "if mtp is not None and not slab_mtp and mtp.rows != chunk_state.rows:" in body
    assert "32-row chunk's option" not in body
    # the extension's rows run after layer 47 with the chunk's selectors (None at 128 rows), before the roots go
    assert body.index("mtp.forward_chunk_rows(") < body.index("_deallocate_unique(residual)")
    # the slab form (2026-09-26): a slab with drafting takes the 128-row twin allocated with the slab's row count and
    # runs it over the slab's residual from the slab body's host-written RoPE rows (the chunk state's persistent rows)
    assert "slab_mtp = mtp is not None and is_slab_rows(chunk_state.rows)" in body
    assert "if slab_mtp and not (mtp.rows == LONG_CHUNK_ROWS and mtp.slab_rows == chunk_state.rows):" in body
    assert body.index("mtp.forward_slab_rows(") < body.index("_deallocate_unique(residual)")
    assert "rope = chunk_state.rope_rows" in body
    assert "mtp.forward_slab_rows(self, residual, rope_rows=rope, position_scalar=state.position.scalar)" in body
    extension = inspect.getsource(mtp_v2.Qwen38TTNNMTPChunkExtension)
    assert "rows: int = CHUNK_ROWS" in extension
    assert "token_row = model.model_io.embedding.upload_token_rows(rows)" in extension
    assert "local_combine_output=chunk_state.local_combine_output," in extension
    assert "if self.rows != CHUNK_ROWS:" in extension  # finish_chunk


# --------------------------------------------------------------------------- the driver's mixed tail with MTP


LONG_TRACE_ID, TRACE_ID = 128, 77


class _DriverModel:
    allocated_context = 4096

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", chunk_state.rows))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", chunk_state.rows, accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context, positions=None, features=None):
        tokens = list(token_ids)
        assert len(tokens) == chunk_state.rows
        contexts = [ple_context]
        for token in tokens:
            contexts.append((2 if contexts[-1] is None else contexts[-1][1], token))
        self.calls.append(("inputs", chunk_state.rows, tuple(tokens)))
        return SimpleNamespace(rows=chunk_state.rows, contexts=tuple(contexts))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        assert prepared.rows == chunk_state.rows

    def finish_prefill(self, state, chunk_state, prefilled: int, *, rope_shift: int = 0) -> None:
        self.calls.append(("finish", chunk_state.rows, prefilled))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False, mtp=None) -> None:
        self.calls.append(("eager", chunk_state.rows, None if mtp is None else mtp.rows))


class _Extension:
    def __init__(self, rows: int, log: list) -> None:
        self.rows = rows
        self.log = log

    def reset_chunk(self) -> None:
        self.log.append(("mtp_reset", self.rows))

    def write_tokens(self, model, token_ids, features=None) -> None:
        assert len(token_ids) == self.rows
        self.log.append(("mtp_tokens", self.rows, tuple(int(token) for token in token_ids)))
        if features is not None:
            self.log.append(("mtp_features", self.rows, int(features.shape[0])))

    def finish_chunk(self, model, *, prefilled: int) -> None:
        self.log.append(("mtp_finish", self.rows, prefilled))


def _driver(monkeypatch, *, traced: bool = True, long_chunks: bool = True):
    model = _DriverModel()
    log = model.calls
    fake_ttnn = SimpleNamespace(
        _ttnn_execute_trace=lambda mesh, trace_id, *, cq_id, blocking: log.append(("replay", trace_id)),
        record_event=lambda mesh, cq_id: "event",
        event_synchronize=lambda event: None,
        synchronize_device=lambda mesh: log.append(("sync",)),
    )
    monkeypatch.setattr(driver_module, "ttnn", fake_ttnn)

    class Tracker:
        @staticmethod
        def verify_before_replay(mesh, trace_id) -> None:
            log.append(("verify", trace_id))

    monkeypatch.setattr(driver_module, "TraceAllocationTracker", Tracker)
    short, long = SimpleNamespace(rows=CHUNK_ROWS), SimpleNamespace(rows=LONG_CHUNK_ROWS)
    mtp, long_mtp = _Extension(CHUNK_ROWS, log), _Extension(LONG_CHUNK_ROWS, log)
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        "mesh",
        "state",
        short,
        TRACE_ID if traced else None,
        forced_step=lambda token, context: (2 if context is None else context[1], token),
        long_chunk_state=long if long_chunks else None,
        long_chunk_trace_id=LONG_TRACE_ID if (traced and long_chunks) else None,
        mtp=mtp,
        long_mtp=long_mtp if long_chunks else None,
    )
    return SimpleNamespace(model=model, log=log, prefill=prefill, mtp=mtp, long_mtp=long_mtp)


def test_driver_runs_the_560_token_prompt_as_four_long_chunks_and_two_short_ones_with_the_mtp_tokens_per_form(
    monkeypatch,
) -> None:
    """559 chunked tokens (the 560-token prompt less the teacher-forced last one) = 4 x 128 + 32 + a padded 15-row
    chunk: the long extension takes 128 tokens per long chunk, the 32-row one 32 per short chunk (the tail padded),
    every MTP token the chunk's row one position ahead and the following token after the last."""

    driver = _driver(monkeypatch)
    count, following_token = 559, 7
    tokens = [1000 + index for index in range(count)]
    result = driver.prefill.run(tokens, start_position=0, ple_context=None, following_token=following_token)
    assert result.position == count
    assert (result.timing.long_chunks, result.timing.chunks, result.timing.tail_rows) == (4, 2, 15)
    log = driver.log
    # the seed: the 32-row state and its extension first, then the 128-row state and its twin, then the verifies
    assert log[:6] == [
        ("sync",),
        ("reset", CHUNK_ROWS),
        ("reset", LONG_CHUNK_ROWS),
        ("mtp_reset", CHUNK_ROWS),
        ("mtp_reset", LONG_CHUNK_ROWS),
        ("verify", TRACE_ID),
    ]
    following = tokens[1:] + [following_token]
    expected_tokens = [(LONG_CHUNK_ROWS, tuple(following[128 * i : 128 * (i + 1)])) for i in range(4)]
    expected_tokens.append((CHUNK_ROWS, tuple(following[512:544])))
    expected_tokens.append((CHUNK_ROWS, tuple(following[544:559]) + (driver_module.CHUNK_PAD_TOKEN_ID,) * 17))
    written = [(entry[1], entry[2]) for entry in log if entry[0] == "mtp_tokens"]
    assert written == expected_tokens
    # every chunk's MTP tokens are written between its inputs and its replay, in plan order
    body = [(entry[0], entry[1]) for entry in log if entry[0] in ("inputs", "mtp_tokens", "replay")]
    assert body == [
        entry
        for rows, trace in [(LONG_CHUNK_ROWS, LONG_TRACE_ID)] * 4 + [(CHUNK_ROWS, TRACE_ID)] * 2
        for entry in (("inputs", rows), ("mtp_tokens", rows), ("replay", trace))
    ]
    # the accept scalar once, on the 32-row state, before the padded tail; the hand-off from the 32-row forms only
    assert [entry for entry in log if entry[0] == "accepted"] == [("accepted", CHUNK_ROWS, 14)]
    assert log[-4:] == [("sync",), ("finish", CHUNK_ROWS, count), ("mtp_finish", CHUNK_ROWS, count), ("sync",)]


@pytest.mark.parametrize("count", (129, 256, 513))
def test_driver_prompts_ending_on_a_128_boundary_write_no_short_chunk_and_hand_off_closed(monkeypatch, count) -> None:
    driver = _driver(monkeypatch)
    tokens = [1000 + index for index in range(count)]
    result = driver.prefill.run(tokens, start_position=0, ple_context=None, following_token=5)
    long_chunks, short = divmod(count, LONG_CHUNK_ROWS)
    expected_short = len(driver_module.chunk_accepts(short))
    assert (result.timing.long_chunks, result.timing.chunks) == (long_chunks, expected_short)
    written = [entry[1] for entry in driver.log if entry[0] == "mtp_tokens"]
    assert written == [LONG_CHUNK_ROWS] * long_chunks + [CHUNK_ROWS] * expected_short
    assert ("finish", CHUNK_ROWS, count) in driver.log and ("mtp_finish", CHUNK_ROWS, count) in driver.log
    assert ("mtp_finish", LONG_CHUNK_ROWS, count) not in driver.log


def test_driver_eager_form_passes_the_extension_of_the_chunk_form(monkeypatch) -> None:
    driver = _driver(monkeypatch, traced=False)
    driver.prefill.run([1000 + index for index in range(160)], start_position=0, ple_context=None, following_token=5)
    assert [entry for entry in driver.log if entry[0] == "eager"] == [
        ("eager", LONG_CHUNK_ROWS, LONG_CHUNK_ROWS),
        ("eager", CHUNK_ROWS, CHUNK_ROWS),
    ]


def test_driver_without_long_chunks_resets_and_writes_the_32_row_extension_only(monkeypatch) -> None:
    driver = _driver(monkeypatch, long_chunks=False)
    driver.prefill.run([1000 + index for index in range(160)], start_position=0, ple_context=None, following_token=5)
    assert [entry for entry in driver.log if entry[0] == "mtp_reset"] == [("mtp_reset", CHUNK_ROWS)]
    assert [entry[1] for entry in driver.log if entry[0] == "mtp_tokens"] == [CHUNK_ROWS] * 5


def test_driver_refuses_mtp_long_chunks_without_the_twin_the_twin_without_its_base_and_slabs_with_mtp(
    expect_error,
) -> None:
    short, long, slab = SimpleNamespace(rows=32), SimpleNamespace(rows=128), SimpleNamespace(rows=2048)
    build = lambda **kwargs: driver_module.Qwen38ChunkPrefill(
        _DriverModel(), "mesh", "state", short, None, forced_step=lambda t, c: c, **kwargs
    )
    with expect_error(ValueError, match="long_mtp"):  # allow-pytest.raises: pure contract test
        build(long_chunk_state=long, mtp=object())
    with expect_error(ValueError, match="32-row extension"):  # allow-pytest.raises: pure contract test
        build(long_chunk_state=long, long_mtp=object())
    with expect_error(ValueError, match="long chunk state"):  # allow-pytest.raises: pure contract test
        build(mtp=object(), long_mtp=object())
    with expect_error(ValueError, match="slab"):  # allow-pytest.raises: pure contract test
        build(long_chunk_state=long, slab_state=slab, mtp=object(), long_mtp=object())
    # the admitted pairs
    build(long_chunk_state=long, mtp=object(), long_mtp=object())
    build(mtp=object())
    build(long_chunk_state=long)


# --------------------------------------------------------------------------- the chain: refusal lifted, twin wired


SESSION_SOURCE = Path(session_module.__file__)
SERVER_SOURCE = SESSION_SOURCE.with_name("qwen38_chat_server.py")


def _opened() -> str:
    source = SESSION_SOURCE.read_text(encoding="utf-8")
    return source[source.index("    def open(") : source.index("    def trace_ids(")]


def test_chain_open_admits_long_chunks_and_a_slab_with_mtp() -> None:
    opened = _opened()
    assert "long chunks and MTP drafting are alternatives" not in opened
    assert "32-row chunk option" not in opened
    assert "a prefill slab and MTP drafting are alternatives" not in opened
    # the slab with drafting: the 128-row twin allocated with the slab form, warmed and captured with the slab body
    assert "model, verify, long_chunk_state, base=chain_mtp.chunk_extension, slab_rows=slab_rows" in opened
    # the chain's live admission (READY's record) counts the twin's slab form: the slab rows reach the call
    assert (
        "                long_chunks=long_chunks,\n"
        "                slab_rows=slab_rows,  # the twin's slab form under --prefill-slab: the default chain's\n"
    ) in opened
    assert "slab_extension.write_slab_tokens(model, [*warm_slab_tokens[1:], warm_slab_tokens[0]])" in opened
    assert "model.forward_prefill_chunk_generic(slab_state, state, mtp=slab_extension)" in opened
    assert (
        'guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat slab {label}"),\n'
        "                cq_id=0,\n"
        "                mtp=None if chain_mtp is None else chain_mtp.long_chunk_extension,"
    ) in opened
    assert "long_chunk_extension" in {field.name for field in dataclasses.fields(session_module.Qwen38ChainMTP)}
    assert session_module.Qwen38ChainMTP.__dataclass_fields__["long_chunk_extension"].default is None


def test_chain_open_allocates_warms_captures_and_marks_the_128_row_twin_in_order() -> None:
    opened = _opened()
    order = (
        "mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk_state)",
        "chain_mtp.long_chunk_extension = mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(",
        "model, verify, long_chunk_state, base=chain_mtp.chunk_extension",
        'mtp_dram_bytes_per_bank["states"] = dram_allocated_per_bank() - allocated_before_mtp_states',
        'marker("before-chat-long-chunk-warm-pass")',
        "long_chunk_extension = None if chain_mtp is None else chain_mtp.long_chunk_extension",
        "chain_mtp.alignment.layer.reset_generic_state_inplace(chain_mtp.alignment.generic_state)",
        "chain_mtp.chunk_extension.reset_chunk()",
        "long_chunk_extension.reset_chunk()",
        "long_chunk_extension.write_tokens(model, [*WARM_LONG_CHUNK_TOKEN_IDS[1:], WARM_LONG_CHUNK_TOKEN_IDS[0]])",
        "model.forward_prefill_chunk_generic(long_chunk_state, state, mtp=long_chunk_extension)",
        'marker("before-chat-chunk-warm-pass")',
        "chain_mtp.long_chunk_extension.reset_chunk()",
        "acknowledge_corruptible(chain_mtp.long_chunk_extension.token_row)",
        "set_misses_allowed(False)",
        "chain.chunk_trace_id = model.capture_prefill_chunk(",
        "dram_after_chunk = dram_allocated_per_bank()",
        "chain.long_chunk_trace_id = model.capture_prefill_chunk(",
        "mtp=None if chain_mtp is None else chain_mtp.long_chunk_extension,",
        "dram_after_long_chunk = dram_allocated_per_bank()",
        'marker("after-chat-slab-capture")',
        "dram_after_prefill_captures = dram_allocated_per_bank()",
        # the MTP captures and the growth gate live in open's per-chain helper (QWEN38_MTP_DRAFTS_PER_REQUEST runs it
        # once per drafting chain), defined after the prefill captures and run at the marker below; the chain's traces
        # grow from the baseline the caller passes -- the last prefill capture's allocation for the default chain
        "def capture_mtp_chain(target: Qwen38ChainMTP, dram_baseline: int) -> int:",
        '"long_chunk_trace": dram_after_long_chunk - dram_after_chunk,',
        '"mtp_fused_traces": dram_after_fused - dram_baseline,',
        '"mtp_traces": dram_after_target - dram_baseline,',
        'if mtp_growth > target.admission["required_free_bytes_per_bank"]:',
        'marker("before-chat-mtp-captures")',
        "dram_after_previous = capture_mtp_chain(chain_mtp, dram_after_prefill_captures)",
    )
    # in order: every fragment after the previous one (the slab warm block, earlier in open, reuses two of them)
    position = -1
    for fragment in order:
        position = opened.index(fragment, position + 1)
    # the slab warm pass and capture with the twin's slab form (2026-09-26), before the long-chunk warm
    slab_order = (
        'marker("before-chat-slab-warm-pass")',
        "slab_extension = None if chain_mtp is None else chain_mtp.long_chunk_extension",
        "slab_extension.write_slab_tokens(model, [*warm_slab_tokens[1:], warm_slab_tokens[0]])",
        "model.forward_prefill_chunk_generic(slab_state, state, mtp=slab_extension)",
        'marker("after-chat-slab-warm-pass")',
        'marker("before-chat-long-chunk-warm-pass")',
    )
    position = -1
    for fragment in slab_order:
        position = opened.index(fragment, position + 1)
    assert opened.index(
        "model, verify, long_chunk_state, base=chain_mtp.chunk_extension, slab_rows=slab_rows"
    ) < opened.index('marker("before-chat-slab-warm-pass")')
    # the growth gate no longer books the 128-row trace as MTP traces
    assert '"mtp_traces": dram_after_mtp - dram_after_chunk,' not in opened
    assert '"mtp_fused_traces": dram_after_fused - dram_after_chunk,' not in opened
    # the live admission's call keeps its keyword order (the routing pin) and gains the long-chunks term last
    assert (
        "                live=dram_free_view(),\n"
        "                verify_forms=len(forms),\n"
        "                long_chunks=long_chunks,\n"
    ) in opened
    # the driver gets both twins; the release order is the twin before the extension it was allocated beside
    source = SESSION_SOURCE.read_text(encoding="utf-8")
    prefill = source[source.index("    def chunk_prefill(") : source.index("    # -- the MTP pass loop primitives")]
    assert "mtp=None if self.mtp is None else self.mtp.chunk_extension," in prefill
    assert "long_mtp=None if self.mtp is None else self.mtp.long_chunk_extension," in prefill
    close = source[source.index("    def close(self)") :]
    assert close.index("self.mtp.long_chunk_extension.release()") < close.index("self.mtp.chunk_extension.release()")
    assert close.index("release_chunk_state(self.long_chunk_state)") < close.index(
        "self.mtp.long_chunk_extension.release()"
    )


def _with_margin(remainder: int) -> int:
    return -(-remainder * (100 + session_module.MTP_GROWTH_ESTIMATE_MARGIN_PERCENT) // 100)


def test_admission_carries_the_long_chunk_terms(expect_error, monkeypatch) -> None:
    """A --long-chunks chain's admission takes the measured 128-row chunk state and trace off the free side (the
    table, and the live after-build read where they are still to come; not the after-captures read) and adds the MTP
    layer's 128-row extension remainder to the states estimate (25,600 bytes per bank, measured 2026-09-26 as the
    states term of the --mtp --long-chunks chain's growth record less the plain --mtp chain's); the default record is
    unchanged.  (The pair bytes are stubbed: the packer's layout tables need the runtime.)"""

    monkeypatch.setattr(session_module, "packed_bf4_bytes_per_device", lambda *, ring_size: (80 << 20, 40 << 20))
    assert session_module.LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES == 1_603_483_392 - 1_569_890_816 == 33_592_576
    extension = session_module.MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK
    assert extension == 15_339_392 - 15_313_792 == 25_600
    states = session_module.MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS[5]
    for context in (32768, 65536, 131072):
        plain = session_module.mtp_capacity_admission(context, drafts=4)
        long = session_module.mtp_capacity_admission(context, drafts=4, long_chunks=True)
        assert (plain["long_chunks"], long["long_chunks"]) == (False, True)
        assert (
            plain["long_chunks_bytes_per_bank_after_captures"] == plain["mtp_long_chunk_extension_bytes_per_bank"] == 0
        )
        assert plain["mtp_growth_remainders_bytes_per_bank"]["long_chunk_extension"] == 0
        assert long["long_chunks_bytes_per_bank_after_captures"] == 33_592_576
        assert long["mtp_long_chunk_extension_bytes_per_bank"] == extension
        assert long["mtp_growth_remainders_bytes_per_bank"]["long_chunk_extension"] == extension
        for key in ("free_bytes_per_bank_after_captures", "largest_contiguous_bytes_free_per_bank_after_captures"):
            assert plain[key] - long[key] == 33_592_576, key
        states_delta = _with_margin(states + extension) - _with_margin(states)
        estimates = plain["mtp_growth_estimate_bytes_per_bank"], long["mtp_growth_estimate_bytes_per_bank"]
        assert estimates[1]["states"] - estimates[0]["states"] == states_delta
        assert long["required_free_bytes_per_bank"] - plain["required_free_bytes_per_bank"] == states_delta
        assert (
            long["headroom_bytes_per_bank"]
            == long["free_bytes_per_bank_after_captures"] - long["required_free_bytes_per_bank"]
        )
        assert long["fits"] == (
            long["free_bytes_per_bank_after_captures"] >= long["required_free_bytes_per_bank"]
            and long["largest_contiguous_bytes_free_per_bank_after_captures"]
            >= long["required_largest_contiguous_bytes_per_bank"]
        )
        assert plain["fits"] and long["fits"], context
    # the live read: after the build the 128-row state and trace are still to come (off the free side); after the
    # captures they are in the reading already (nothing off)
    view = {"num_banks": 8, "free_bytes_per_bank": 1_500 << 20, "largest_contiguous_bytes_free_per_bank": 1_400 << 20}
    for point, delta in (("after_build", 33_592_576), ("after_captures", 0)):
        plain = session_module.mtp_capacity_admission(32768, drafts=4, live=view, live_point=point)
        long = session_module.mtp_capacity_admission(32768, drafts=4, live=view, live_point=point, long_chunks=True)
        assert plain["free_bytes_per_bank_after_captures"] - long["free_bytes_per_bank_after_captures"] == delta, point
        assert long["long_chunks_bytes_per_bank_after_captures"] == 33_592_576
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        session_module.mtp_capacity_admission(32768, long_chunks=1)


def test_admission_carries_the_slab_form_term(expect_error, monkeypatch) -> None:
    """A --prefill-slab chain under --mtp adds the twin's slab form to the states estimate: per 128-row slice one FP32
    token tile (4,096 bytes per bank) and per slice after the first one 32-byte offset page (66,016 bytes per bank at
    2048 rows); the form takes a slab row count with the long chunks and nothing else changes against the
    --long-chunks record."""

    monkeypatch.setattr(session_module, "packed_bf4_bytes_per_device", lambda *, ring_size: (80 << 20, 40 << 20))
    assert session_module.MTP_SLAB_TOKEN_TILE_BYTES == 4_096 and session_module.MTP_SLAB_OFFSET_PAGE_BYTES == 32
    assert session_module.mtp_slab_form_bytes_per_bank(2048) == 16 * 4_096 + 15 * 32 == 66_016
    states = session_module.MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS[5]
    extension = session_module.MTP_LONG_CHUNK_EXTENSION_BYTES_PER_BANK
    for context in (32768, 65536):
        long = session_module.mtp_capacity_admission(context, drafts=4, long_chunks=True)
        slab = session_module.mtp_capacity_admission(context, drafts=4, long_chunks=True, slab_rows=2048)
        assert (long["mtp_slab_form_bytes_per_bank"], long["slab_rows"]) == (0, None)
        assert (slab["mtp_slab_form_bytes_per_bank"], slab["slab_rows"]) == (66_016, 2048)
        assert slab["mtp_growth_remainders_bytes_per_bank"]["long_chunk_extension"] == extension + 66_016
        delta = _with_margin(states + extension + 66_016) - _with_margin(states + extension)
        assert slab["required_free_bytes_per_bank"] - long["required_free_bytes_per_bank"] == delta
        assert slab["free_bytes_per_bank_after_captures"] == long["free_bytes_per_bank_after_captures"]
        assert slab["fits"], context
    for bad in ({"slab_rows": 2048}, {"long_chunks": True, "slab_rows": 128}):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            session_module.mtp_capacity_admission(32768, drafts=4, **bad)


def test_server_admits_long_chunks_with_mtp_and_reports_the_twin() -> None:
    server = SERVER_SOURCE.read_text(encoding="utf-8")
    assert "--long-chunks and --mtp are alternatives" not in server
    assert "--prefill-slab and --mtp are alternatives" not in server  # the slab runs the twin's slab form (2026-09-26)
    assert (
        "        else mtp_capacity_admission(\n"
        "            resident_context.allocated_context,\n"
        "            drafts=args.mtp,\n"
        "            verify_forms=len(forms),\n"
        "            long_chunks=bool(args.long_chunks) or args.prefill_slab is not None,\n"
        "            moe_rows=mtp_moe_rows,\n"  # the verify MoE row count knob (QWEN38_MTP_MOE_ROWS) sits beside it
        "            gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),\n"
        "            slab_rows=args.prefill_slab,\n"
        "            # the tower's MODELED terms (the image path is the default path; the live decision is the warm hook's)\n"
        '            vision_resident_bytes_per_bank=-(-vision_resident_layout(mesh_size=4)["device_total"] // 8),\n'
        "            vision_peak_activation_bytes_per_bank=-(\n"
        "                -VISION_ROW_BUCKETS[-1] * PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE // 8\n"
        "            ),\n"
        "        )"
    ) in server
    assert '"long_chunk_extension": chain.mtp.long_chunk_extension is not None,' in server
