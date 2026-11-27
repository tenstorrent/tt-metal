# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the K3.1 FP64 chain-golden reader on a small registered chain: rows across segment and file
boundaries, sampled rows below a bound, refusal of products or rows the golden does not hold and of a missing or
other-version chain, and the per-layer weight set and digest. CPU only."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    GoldenMismatch,
    GoldenNotFound,
    product_schema,
    write_manifest,
    write_token_ids,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_chain_golden as chain

ROWS, SEGMENT = 10, 4  # segments [0, 4) [4, 8) [8, 10)


def _write_chain(root, golden_id="k31-test-chain", producer_version=chain.PRODUCER_VERSION):
    """A registered chain golden: row r's ``moe_output_1`` is ``[r, -r]``, its key ``latent`` ``[10 r]``, and every
    segment samples its first row (``attn_output_3`` = ``[r + 0.5]``)."""
    directory = root / golden_id
    directory.mkdir(parents=True)
    segments = []
    for i, start in enumerate(range(0, ROWS, SEGMENT)):
        r = torch.arange(start, min(start + SEGMENT, ROWS), dtype=torch.float32)
        payload = {
            "start": start,
            "end": start + r.numel(),
            "moe_output_1": torch.stack([r, -r], 1),
            "sampled_rows": torch.tensor([start]),
            "attn_output_3": torch.tensor([[start + 0.5]], dtype=torch.float64),
        }
        state = {"keys": {"latent": (10 * r).double().unsqueeze(1)}, "kda": {}}
        torch.save(payload, directory / f"segment_{i:06d}.pt")
        torch.save(state, directory / f"state_{i:06d}.pt")
        segments.append({"start": start, "end": start + r.numel()})
    digest = write_token_ids(directory, torch.arange(ROWS))
    products = {
        "moe_output_1": product_schema("row", payload["moe_output_1"]),
        "latent": product_schema("row", state["keys"]["latent"]),
        "sampled_rows": product_schema("sampled_row", payload["sampled_rows"]),
        "attn_output_3": product_schema("sampled_row", payload["attn_output_3"]),
    }
    write_manifest(
        directory,
        {
            "golden_id": golden_id,
            "producer": chain.PRODUCER,
            "producer_version": producer_version,
            "identity": {"model": chain.MODEL, "token_source": "test", "weights_sha256": {}},
            "token_ids_sha256": digest,
            "rows": ROWS,
            "segment": SEGMENT,
            "segments": segments,
            "products": products,
        },
    )


def test_read_rows_spans_segments_and_files(tmp_path) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    got = chain.read_rows(golden, ["moe_output_1", "latent"], 3, 9)
    assert torch.equal(got["moe_output_1"], torch.tensor([[3.0, -3], [4, -4], [5, -5], [6, -6], [7, -7], [8, -8]]))
    assert torch.equal(got["latent"], torch.tensor([[30.0], [40], [50], [60], [70], [80]], dtype=torch.float64))
    assert torch.equal(chain.read_rows(golden, ["moe_output_1"], 9, 10)["moe_output_1"], torch.tensor([[9.0, -9]]))


def test_read_rows_refuses_what_the_golden_does_not_hold(tmp_path, expect_error) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    with expect_error(ValueError, "no per-row products"):
        chain.read_rows(golden, ["attn_output_3"], 0, 4)
    with expect_error(ValueError, "no per-row products"):
        chain.read_rows(golden, ["moe_output_2"], 0, 4)
    with expect_error(ValueError, "not inside"):
        chain.read_rows(golden, ["moe_output_1"], 8, 11)


def test_read_sampled_keeps_rows_below_end(tmp_path) -> None:
    _write_chain(tmp_path)
    golden = chain.open_k31_chain("k31-test-chain", tmp_path)
    every = chain.read_sampled(golden)
    assert every["sampled_rows"].tolist() == [0, 4, 8]
    assert every["attn_output_3"].flatten().tolist() == [0.5, 4.5, 8.5]
    assert chain.read_sampled(golden, 8)["sampled_rows"].tolist() == [0, 4]  # (row 8 is not below 8)


def test_open_fails_on_a_missing_or_other_version_chain(tmp_path, expect_error) -> None:
    with expect_error(GoldenNotFound, "no golden"):
        chain.open_k31_chain("k31-test-chain", tmp_path)
    _write_chain(tmp_path, producer_version=chain.PRODUCER_VERSION + 1)
    with expect_error(GoldenMismatch, "producer_version"):
        chain.open_k31_chain("k31-test-chain", tmp_path)


def test_layer_weights_reads_the_chain_tensor_set(tmp_path, expect_error) -> None:
    """Layer 3 (DSA): the AttnRes, router and DSA weights, under the checkpoint's key root, digested by name; no
    other layer's set is supported."""
    names = chain.ATTN_RES_WEIGHTS + chain.ROUTER_WEIGHTS + chain.DSA_WEIGHTS
    assert len(set(names)) == len(names) == 6 + 2 + 13
    tensors = {f"model.layers.3.{n}": torch.full((2,), float(i), dtype=torch.bfloat16) for i, n in enumerate(names)}
    tensors["model.layers.4.input_layernorm.weight"] = torch.zeros(2)  # (another layer's: not read)
    save_file(tensors, tmp_path / "shard.safetensors")
    weight_map = {k: "shard.safetensors" for k in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    got, digest = chain.layer_weights(tmp_path, 3)
    assert sorted(got) == sorted(names)
    assert torch.equal(got["self_attn.o_proj.weight"], tensors["model.layers.3.self_attn.o_proj.weight"])
    assert digest == {"layers.3": chain.tensor_digest(got)}
    got["self_attn.o_proj.weight"] = got["self_attn.o_proj.weight"] + 1
    assert chain.tensor_digest(got) != digest["layers.3"]
    with expect_error(ValueError, "DSA layer 3"):
        chain.layer_weights(tmp_path, 2)
