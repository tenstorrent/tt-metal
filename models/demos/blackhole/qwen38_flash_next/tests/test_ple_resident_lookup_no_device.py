# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The PLE on the CPU: the host n-gram lookup and the post-lookup layer on the released checkpoint (against the pinned
Transformers ``Qwen4Exp`` when ``QWEN38_TRANSFORMERS_SRC`` is set); persistent-descriptor row reads that must return
the host oracle's bytes; and the decode-step form of the lookup: the scalar-integer n-gram hash against
``reference.ngram_token_ids`` over the real id ranges and the row payload against the host oracle.
"""

import functools
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint, Qwen38CheckpointRowReader
from models.demos.blackhole.qwen38_flash_next.reference import build_ngram_hash_spec, ngram_token_ids
from models.demos.blackhole.qwen38_flash_next.tests.helpers.model import qwen4_exp_transformers, requires_transformers
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding, Qwen38PLE, Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38ResidentPLELookup, ngram_token_ids_decode_step

PARTS = 4
ROWS_PER_PART = 1024
HEAD_DIM = 8
EOS = 999
# The pinned model's hash geometry (config.py pins every field): 248,320 unigram
# ids, 16 heads over primes above 19,999,999, EOS 248044.
REAL_VOCAB_SIZE = 248_320
REAL_EOS = 248_044
REAL_SPEC = build_ngram_hash_spec(
    unigram_vocab_size=REAL_VOCAB_SIZE,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    ple_layer_index=0,
    seed=1234,
    divisible_by=128,
)
# A 33-token chain from a device run of the per-position sequential mode.
CHAIN_33 = (17, 15, 16, 22, 95859, 16, 17, 96212, 3709, 101857, 97191, 98478, 95759, 26076, 98817, 97035, 123772)
CHAIN_33 += (9616, 17, 15, 16, 22, 12, 17, 15, 18, 20, 95859, 7313, 24273, 10992, 271, 107680)


# --------------------------------------------------------------------------- the host lookup and the PLE layer


@pytest.fixture(scope="module")
def embedding(release_checkpoint):
    return Qwen38HostPLEEmbedding(release_checkpoint)


@pytest.fixture(scope="module")
def weights(release_checkpoint):
    return Qwen38PLEWeights.from_checkpoint(release_checkpoint)


def test_random_access_rows_match_safetensors_slice_without_materializing_table(release_checkpoint):
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    indices = torch.tensor([[0, 17], [2_500_010, 17]])
    got = release_checkpoint.tensor_rows(name, indices)
    expected = torch.stack(
        [release_checkpoint.tensor_slice(name, (int(index), slice(None))) for index in indices.flatten()]
    ).reshape(2, 2, 160)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_exact_host_ngram_lookup_is_split_invariant_and_shards_640_per_device(embedding):
    tokens = torch.tensor([[17, 29, 31, 248044, 43, 47]])
    full, full_context = embedding.lookup(tokens)
    first, first_context = embedding.lookup(tokens[:, :4])
    second, second_context = embedding.lookup(tokens[:, 4:], first_context)

    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=0, atol=0)
    torch.testing.assert_close(second_context, full_context, rtol=0, atol=0)
    assert full.shape == (1, 6, 2560)
    device_parts = embedding.shard_result(full)
    assert len(device_parts) == 4
    assert all(part.shape == (1, 6, 640) for part in device_parts)
    torch.testing.assert_close(torch.cat(device_parts, dim=-1), full, rtol=0, atol=0)


@requires_transformers
def test_exact_ple_post_lookup_matches_pinned_transformers(embedding, weights, checkpoint_dir):
    configuration, modeling = qwen4_exp_transformers()

    # Instantiate the official post-lookup path with a tiny temporary hash
    # vocabulary, then replace its embedding module.  This avoids allocating
    # the checkpoint's 51.2B host-resident n-gram parameters in the oracle.
    config = configuration.Qwen4ExpTextConfig.from_pretrained(checkpoint_dir)
    config.ngram_vocab_size_base = 101
    oracle = modeling.Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).to(torch.bfloat16).eval()

    tokens = torch.tensor([[101, 202, 303]])
    exact_embeddings, _ = embedding.lookup(tokens)

    class FixedEmbedding(nn.Module):
        def forward(self, input_ids, past_key_values):
            assert torch.equal(input_ids, tokens)
            assert past_key_values is None
            return exact_embeddings

    oracle.ple_embedding = FixedEmbedding()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)

    torch.manual_seed(229)
    hidden = torch.randn(1, 3, 10240, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = oracle(hidden, tokens, past_key_values=None)
        got, state = Qwen38PLE(embedding, weights).forward(hidden, tokens)

    torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
    assert state.token_context.shape == (1, 2)
    assert state.conv.shape == (1, 10240, 9)


def test_exact_ple_prefill_equals_tokenwise_state_transition(embedding, weights):
    torch.manual_seed(233)
    tokens = torch.tensor([[501, 502, 503]])
    hidden = torch.randn(1, 3, 10240, dtype=torch.bfloat16)
    component = Qwen38PLE(embedding, weights)

    full_output, full_state = component.forward(hidden, tokens)
    state = None
    pieces = []
    for position in range(tokens.shape[1]):
        output, state = component.forward(hidden[:, position : position + 1], tokens[:, position : position + 1], state)
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.token_context, full_state.token_context, rtol=0, atol=0)
    torch.testing.assert_close(state.conv, full_state.conv, rtol=0.01, atol=1e-3)


# --------------------------------------------------------------------------- the resident lookup


class _SyntheticCheckpoint(Qwen38Checkpoint):
    """Real header/guard/pread code over a tiny corpus without the pinned index."""

    def __init__(self, root: Path, weight_map: dict[str, str], file_guard=None) -> None:
        self.root = Path(root).resolve()
        self._file_guard = file_guard
        self.weight_map = dict(weight_map)
        self.shards = tuple(sorted(set(weight_map.values())))


def _table_name(part: int) -> str:
    return f"table.shard_{part}.weight"


def _synthetic_checkpoint(root: Path, *, parts_per_file: int = 1, file_guard=None) -> _SyntheticCheckpoint:
    """``PARTS`` table parts over ``PARTS / parts_per_file`` shard files (the real table: 128 parts over 33 files)."""

    torch.manual_seed(1)
    weight_map = {}
    for first in range(0, PARTS, parts_per_file):
        shard = f"part-{first}.safetensors"
        # The leading tensor gives the table a nonzero data offset inside the file.
        tensors = {f"leading_{first}.bias": torch.randn(37, dtype=torch.bfloat16)}
        weight_map[f"leading_{first}.bias"] = shard
        for part in range(first, min(first + parts_per_file, PARTS)):
            tensors[_table_name(part)] = torch.randn(ROWS_PER_PART, HEAD_DIM, dtype=torch.bfloat16)
            weight_map[_table_name(part)] = shard
        save_file(tensors, str(root / shard))
    return _SyntheticCheckpoint(root, weight_map, file_guard)


def _synthetic_host_embedding(checkpoint: Qwen38Checkpoint) -> SimpleNamespace:
    spec = build_ngram_hash_spec(
        unigram_vocab_size=1000,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=101,
        ple_layer_index=0,
        seed=1234,
        divisible_by=PARTS * ROWS_PER_PART // 2,
    )
    assert spec.padded_vocab_size == PARTS * ROWS_PER_PART
    return SimpleNamespace(
        checkpoint=checkpoint,
        table_names=tuple(_table_name(part) for part in range(PARTS)),
        rows_per_shard=ROWS_PER_PART,
        embedding_head_dim=HEAD_DIM,
        spec=spec,
        config=SimpleNamespace(eos_token_id=EOS, vocab_size=1000),
    )


def _assert_bitwise_equal(got: torch.Tensor, expected: torch.Tensor) -> None:
    assert got.dtype == expected.dtype == torch.bfloat16
    assert got.shape == expected.shape
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def _assert_lookups_are_the_oracles(resident, oracle_lookup, tokens) -> None:
    """The prefill lookup of ``tokens`` and its one-token decode chain return the oracle's rows and contexts bitwise."""

    expected, expected_context = oracle_lookup(tokens)
    got, got_context = resident.lookup(tokens)
    _assert_bitwise_equal(got, expected)
    assert torch.equal(got_context, expected_context)
    context = oracle_context = None
    for position in range(tokens.shape[1]):
        step = tokens[:, position : position + 1]
        got, context = resident.lookup(step, context)
        expected, oracle_context = oracle_lookup(step, oracle_context)
        _assert_bitwise_equal(got, expected)
        assert torch.equal(context, oracle_context)


def test_row_reader_matches_tensor_rows_bitwise_after_one_guarded_open(expect_error, tmp_path) -> None:
    events = []
    checkpoint = _synthetic_checkpoint(tmp_path, file_guard=lambda path, phase: events.append(phase))
    name = _table_name(2)
    reader = Qwen38CheckpointRowReader(checkpoint, name)
    assert [phase for phase in events if "tensor-rows" in phase] == [
        f"before-tensor-rows-open:{name}",
        f"after-tensor-rows-open:{name}",
    ]
    assert (reader.rows, reader.row_shape, reader.row_bytes) == (ROWS_PER_PART, (HEAD_DIM,), 2 * HEAD_DIM)

    torch.manual_seed(2)
    indices = torch.randint(0, ROWS_PER_PART, (3, 5), dtype=torch.long)
    indices[0, 1] = indices[0, 0]
    indices[1, 2] = 0
    indices[2, 4] = ROWS_PER_PART - 1
    events.clear()
    got = reader.read(indices)
    assert not [phase for phase in events if "tensor-rows" in phase]
    _assert_bitwise_equal(got, checkpoint.tensor_rows(name, indices))
    _assert_bitwise_equal(
        reader.read(torch.tensor([7], dtype=torch.long)), checkpoint.tensor_rows(name, torch.tensor([7]))
    )
    assert reader.read(torch.zeros((0,), dtype=torch.long)).shape == (0, HEAD_DIM)
    assert reader.read_rows([5, 5, 9]) == [reader.read_rows([5])[0]] * 2 + reader.read_rows([9])

    with expect_error(IndexError, match="outside"):
        reader.read_rows([ROWS_PER_PART])
    with expect_error(ValueError, match="torch.long"):
        reader.read(torch.tensor([1], dtype=torch.int32))
    reader.close()
    reader.close()
    with expect_error(RuntimeError, match="closed"):
        reader.read_rows([0])


def test_row_reader_rejects_non_bf16_and_scalar_tensors(expect_error, tmp_path) -> None:
    checkpoint = _synthetic_checkpoint(tmp_path)
    with expect_error(ValueError, match="rank>=2 BF16"):
        Qwen38CheckpointRowReader(checkpoint, "leading_0.bias")


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="procfs descriptor links are Linux only")
def test_row_reader_opens_the_guard_supplied_procfs_duplicate(tmp_path) -> None:
    checkpoint = _synthetic_checkpoint(
        tmp_path, file_guard=lambda path, phase: Path(f"/proc/self/fd/{admitted}") if "before" in phase else None
    )
    admitted = os.open(tmp_path / "part-0.safetensors", os.O_RDONLY | os.O_CLOEXEC)
    try:
        reader = Qwen38CheckpointRowReader(checkpoint, _table_name(0))
        indices = torch.tensor([[1, 2], [3, 1]])
        _assert_bitwise_equal(reader.read(indices), checkpoint.tensor_rows(_table_name(0), indices))
        reader.close()
    finally:
        os.close(admitted)


def test_row_reader_fails_closed_when_the_admitted_inode_changes(expect_error, tmp_path) -> None:
    checkpoint = _synthetic_checkpoint(tmp_path)
    replaced = Qwen38CheckpointRowReader(checkpoint, _table_name(1))
    rewritten = Qwen38CheckpointRowReader(checkpoint, _table_name(3))
    assert replaced.read_rows([3, 4]) == replaced.read_rows([3, 4])

    # Replacing the path unlinks the admitted inode (link count and ctime
    # change), the same drift the gate's revalidate_identities refuses.
    shard = tmp_path / "part-1.safetensors"
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"\0" * shard.stat().st_size)
    os.replace(replacement, shard)
    with expect_error(RuntimeError, match="identity drifted"):
        replaced.read_rows([3, 4])

    # Writing into the admitted inode changes its identity and is refused.
    time.sleep(0.05)
    with open(tmp_path / "part-3.safetensors", "r+b") as stream:
        stream.seek(8)
        stream.write(b"!")
    with expect_error(RuntimeError, match="identity drifted"):
        rewritten.read_rows([0])
    with expect_error(RuntimeError, match="identity drifted"):
        rewritten.advise([0])
    replaced.close()
    rewritten.close()


def test_resident_lookup_matches_the_host_oracle_bitwise(expect_error, tmp_path) -> None:
    events = []
    checkpoint = _synthetic_checkpoint(tmp_path, file_guard=lambda path, phase: events.append(phase))
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    assert len(resident.readers) == PARTS
    assert sum("tensor-rows-open" in phase for phase in events) == 2 * PARTS

    tokens = torch.tensor([[17, 29, 31, EOS, 43, 47, 0, 998]])
    events.clear()
    assert resident.lookup(tokens)[0].shape == (1, tokens.shape[1], 16 * HEAD_DIM)
    assert not events
    _assert_lookups_are_the_oracles(resident, functools.partial(Qwen38HostPLEEmbedding.lookup, host), tokens)
    with expect_error(ValueError, match="host-resident"):
        resident.lookup(tokens.to("meta"))
    resident.close()


def _scalar_ids(token_id: int, context, *, eos: int, spec):
    return ngram_token_ids_decode_step(
        token_id,
        context,
        eos_token_id=eos,
        multipliers=tuple(spec.layer_multipliers.tolist()),
        head_vocab_sizes=tuple(spec.head_vocab_sizes.tolist()),
        head_offsets=tuple(spec.head_offsets.tolist()),
    )


def test_decode_step_hash_matches_the_reference_bitwise_over_the_real_id_ranges() -> None:
    # 8 EOS patterns over (c0, c1, x), 12,500 random triples each, plus every
    # boundary id in every slot; the reference is evaluated as one batch of
    # single-token steps with explicit contexts.
    generator = torch.Generator().manual_seed(20260902)
    boundary = torch.tensor([0, 1, REAL_EOS - 1, REAL_EOS, REAL_EOS + 1, REAL_VOCAB_SIZE - 2, REAL_VOCAB_SIZE - 1])
    triples = []
    for pattern in range(8):
        sample = torch.randint(0, REAL_VOCAB_SIZE, (12_500, 3), generator=generator)
        for slot in range(3):
            if pattern >> slot & 1:
                sample[:, slot] = REAL_EOS
        triples.append(sample)
    triples.append(torch.cartesian_prod(boundary, boundary, boundary))
    triples = torch.cat(triples)
    assert triples.shape[0] == 100_000 + 7**3
    contexts = triples[:, :2].contiguous()
    tokens = triples[:, 2:3].contiguous()
    expected_ids, expected_context = ngram_token_ids(tokens, contexts, REAL_EOS, REAL_SPEC)
    assert expected_ids.shape == (triples.shape[0], 1, 16)
    padded = REAL_SPEC.padded_vocab_size
    for index, (c0, c1, x) in enumerate(triples.tolist()):
        ids, context = _scalar_ids(x, (c0, c1), eos=REAL_EOS, spec=REAL_SPEC)
        assert ids == expected_ids[index, 0].tolist(), (c0, c1, x)
        assert context == tuple(expected_context[index].tolist()) == (c1, x)
        assert all(0 <= value < padded for value in ids)
    # A fresh history: None and the explicit (eos, eos) context are the same step.
    for x in (0, 17, REAL_EOS, REAL_VOCAB_SIZE - 1):
        fresh_ids, fresh_context = ngram_token_ids(torch.tensor([[x]]), None, REAL_EOS, REAL_SPEC)
        assert _scalar_ids(x, None, eos=REAL_EOS, spec=REAL_SPEC) == (fresh_ids[0, 0].tolist(), (REAL_EOS, x))
        assert _scalar_ids(x, (REAL_EOS, REAL_EOS), eos=REAL_EOS, spec=REAL_SPEC)[0] == fresh_ids[0, 0].tolist()
        assert fresh_context.tolist() == [[REAL_EOS, x]]
    # Head partition: bigram heads never see c0 (only c1 and x), trigram heads do.
    a, _ = _scalar_ids(17, (5, 9), eos=REAL_EOS, spec=REAL_SPEC)
    b, _ = _scalar_ids(17, (6, 9), eos=REAL_EOS, spec=REAL_SPEC)
    assert a[:8] == b[:8] and a[8:] != b[8:]
    # No product may wrap int64: the largest token times the largest multiplier stays below 2**63.
    assert (REAL_VOCAB_SIZE - 1) * max(REAL_SPEC.layer_multipliers.tolist()) < 1 << 63


def test_decode_step_hash_threads_the_context_through_the_33_token_chain() -> None:
    reference_context = None
    context = None
    for token in CHAIN_33:
        expected_ids, reference_context = ngram_token_ids(
            torch.tensor([[token]]), reference_context, REAL_EOS, REAL_SPEC
        )
        ids, context = _scalar_ids(token, context, eos=REAL_EOS, spec=REAL_SPEC)
        assert ids == expected_ids[0, 0].tolist()
        assert list(context) == reference_context[0].tolist()


def test_lookup_token_returns_the_oracle_rows_and_context(expect_error, tmp_path) -> None:
    checkpoint = _synthetic_checkpoint(tmp_path)
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    assert resident.vocab_size == 1000 and resident.eos_token_id == EOS
    assert len(resident.multipliers) == 3 and len(resident.head_vocab_sizes) == len(resident.head_offsets) == 16
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, 1000, (2_000,), generator=generator)
    tokens[::37] = EOS
    tokens[1::53] = EOS
    context = oracle_context = None
    for token in tokens.tolist():
        payload, context = resident.lookup_token(token, context)
        expected, oracle_context = Qwen38HostPLEEmbedding.lookup(host, torch.tensor([[token]]), oracle_context)
        assert isinstance(payload, bytearray) and len(payload) == 16 * HEAD_DIM * 2
        assert torch.equal(
            torch.frombuffer(payload, dtype=torch.bfloat16).view(torch.int16),
            expected.reshape(-1).contiguous().view(torch.int16),
        )
        assert list(context) == oracle_context[0].tolist()
    for bad in (-1, 1000, 1.0, True, EOS + 1):
        with expect_error((ValueError, TypeError)):
            resident.lookup_token(bad, None)
    with expect_error(ValueError, match="exact integers"):
        resident.lookup_token(3, (1000, 2))
    # The stream form (the chunk drivers' host_rows): one batched read, the per-token payloads concatenated in order
    # and the contexts chained as the steps chain them, from a fresh history and from a mid-stream context.
    stream = tokens.tolist()
    for start_context in (None, (EOS, EOS), (5, EOS), (EOS, 9), (11, 12)):
        payloads = []
        contexts = [start_context]
        context = start_context
        for token in stream:
            payload, context = resident.lookup_token(token, context)
            payloads.append(bytes(payload))
            contexts.append(context)
        batched, batched_contexts = resident.lookup_tokens(stream, start_context)
        assert isinstance(batched, bytearray) and bytes(batched) == b"".join(payloads)
        assert batched_contexts == tuple(contexts) and len(batched_contexts) == len(stream) + 1
    with expect_error(ValueError, match="exact integers"):
        resident.lookup_tokens([3, 1000], None)
    with expect_error(ValueError, match="exact integers"):
        resident.lookup_tokens([3], (1000, 2))
    resident.close()


def test_resident_lookup_matches_the_real_checkpoint_oracle_over_a_decode_chain(embedding) -> None:
    resident = Qwen38ResidentPLELookup(embedding)
    tokens = torch.tensor([[17, 15, 16, 29, 31, 248044, 43, 47, 101, 202, 303, 501, 502, 503, 1000, 247_999]])
    _assert_lookups_are_the_oracles(resident, embedding.lookup, tokens)
    resident.close()


# --------------------------------------------------------------------------- the batch's syscall contract


class _SyscallLog:
    """Records every ``os.fstat`` / ``os.posix_fadvise`` / ``os.pread`` the reader issues, in order."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple[str, int]] = []
        for name in ("fstat", "posix_fadvise", "pread"):
            real = getattr(os, name)

            def logged(*args, _name=name, _real=real):
                self.calls.append((_name, args[0]))
                return _real(*args)

            monkeypatch.setattr(os, name, logged)

    def count(self, name: str) -> int:
        return sum(1 for call in self.calls if call[0] == name)


def _hashed_rows(host, tokens, context=None) -> list[int]:
    hashed = []
    for token in tokens:
        ids, context = _scalar_ids(token, context, eos=EOS, spec=host.spec)
        hashed += ids
    return hashed


@pytest.mark.parametrize("parts_per_file", (1, 2))
def test_resident_batch_proves_each_touched_file_once_and_reads_each_distinct_row_once(
    tmp_path, monkeypatch, parts_per_file
):
    """One MTP pass's rows through ``_read_rows``: one ``fstat`` per touched shard file before and after the batch,
    one WILLNEED per distinct row before any row is read, then one ``pread`` each; the bytes are the per-part checked
    reads' bytes in request order, duplicates included."""

    checkpoint = _synthetic_checkpoint(tmp_path, parts_per_file=parts_per_file)
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    assert len({reader.shard for reader in resident.readers}) == PARTS // parts_per_file
    tokens = [17, 29, 31, EOS, 43]  # a k = 4 pass: 5 tokens, 80 hashed rows
    hashed = _hashed_rows(host, tokens)
    distinct = list(dict.fromkeys(hashed))
    parts = {index // ROWS_PER_PART for index in distinct}
    files = {resident.readers[part].shard for part in parts}
    assert len(hashed) == 16 * len(tokens) and len(distinct) < len(hashed) or len(distinct) == len(hashed)

    # The reference bytes: the per-part checked calls, the same rows in the same order.
    expected = bytearray()
    checked = {}
    for part in sorted(parts):
        rows = [index % ROWS_PER_PART for index in distinct if index // ROWS_PER_PART == part]
        for row, payload in zip(rows, resident.readers[part].read_rows(rows)):
            checked[part * ROWS_PER_PART + row] = payload
    for index in hashed:
        expected += checked[index]

    log = _SyscallLog(monkeypatch)
    payload, contexts = resident.lookup_tokens(tokens, None)
    assert bytes(payload) == bytes(expected)
    assert len(contexts) == len(tokens) + 1 and contexts[0] is None
    assert log.count("fstat") == 2 * len(files), log.calls
    assert log.count("posix_fadvise") == len(distinct) and log.count("pread") == len(distinct)
    kinds = [name for name, _ in log.calls]
    first_pread, last_fadvise = kinds.index("pread"), len(kinds) - 1 - kinds[::-1].index("posix_fadvise")
    assert last_fadvise < first_pread, "every WILLNEED is issued before the first pread"
    assert kinds[: len(files)] == ["fstat"] * len(files) and kinds[-len(files) :] == ["fstat"] * len(files)
    # The proof is per file: every touched file's descriptor is proved, none twice per side.
    proved = [descriptor for name, descriptor in log.calls if name == "fstat"]
    assert len(set(proved[: len(files)])) == len(files) and proved[: len(files)] == proved[len(files) :]
    resident.close()


def test_resident_batch_fails_closed_when_a_touched_file_drifts_and_reads_the_others(expect_error, tmp_path) -> None:
    """The per-file proof keeps the fail-closed contract: a batch touching a rewritten file is refused before any
    read (and a rewrite between the proofs would be refused after); a batch over the untouched files still reads."""

    checkpoint = _synthetic_checkpoint(tmp_path, parts_per_file=2)
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    rows_file_0 = [1, 5, ROWS_PER_PART + 7]  # parts 0 and 1 live in part-0.safetensors
    rows_file_1 = [2 * ROWS_PER_PART + 3, 3 * ROWS_PER_PART + 9]  # parts 2 and 3 in part-2.safetensors
    before_0, before_1 = bytes(resident._read_rows(rows_file_0)), bytes(resident._read_rows(rows_file_1))
    time.sleep(0.05)
    with open(tmp_path / "part-0.safetensors", "r+b") as stream:
        stream.seek(8)
        stream.write(b"!")
    with expect_error(RuntimeError, match="identity drifted before"):
        resident._read_rows(rows_file_0)
    with expect_error(RuntimeError, match="identity drifted before"):
        resident._read_rows(rows_file_0 + rows_file_1)  # one drifted file refuses the whole batch
    assert bytes(resident._read_rows(rows_file_1)) == before_1  # the other file's identity stands
    assert before_0 != before_1
    resident.close()
    with expect_error(RuntimeError, match="closed"):
        resident._read_rows(rows_file_1)


def test_row_reader_proven_forms_are_the_checked_forms_without_the_fstat(expect_error, tmp_path, monkeypatch) -> None:
    checkpoint = _synthetic_checkpoint(tmp_path)
    reader = Qwen38CheckpointRowReader(checkpoint, _table_name(1))
    assert reader.shard == "part-1.safetensors"
    rows = [3, 3, 8, 0]
    log = _SyscallLog(monkeypatch)
    checked = reader.read_rows(rows)
    assert (log.count("fstat"), log.count("pread")) == (2, 3)
    log.calls.clear()
    assert reader.read_rows_proven(rows) == checked
    assert (log.count("fstat"), log.count("pread")) == (0, 3)
    log.calls.clear()
    reader.prove_identity("before")
    reader.advise_proven(rows)
    assert (log.count("fstat"), log.count("posix_fadvise"), log.count("pread")) == (1, 3, 0)
    reader.close()
    with expect_error(RuntimeError, match="closed"):
        reader.read_rows_proven([0])
    with expect_error(RuntimeError, match="closed"):
        reader.prove_identity("before")
