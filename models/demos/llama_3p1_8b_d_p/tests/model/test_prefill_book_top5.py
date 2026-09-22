# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 32-layer book-continuation top-5 gate against independent CPU/HF goldens."""

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures/book"
CHECKPOINT = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))


# The two supported lengths share checkpoint files; hash each only once per run.
@lru_cache(maxsize=None)
def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Validate small committed inputs/goldens before allocating any device resources.
def _load_book_golden(context_length):
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    inputs = provenance["input"]
    for filename_key, digest_key in (("text_file", "text_sha256"), ("token_ids_file", "token_ids_sha256")):
        assert _sha256(FIXTURES / inputs[filename_key]) == inputs[digest_key]
    ids = json.loads((FIXTURES / inputs["token_ids_file"]).read_text())
    assert len(ids) == 4096 and ids[0] == 128000 and ids.count(128000) == 1
    assert all(type(token) is int and 0 <= token < 128256 for token in ids)
    ids = ids[:context_length]
    golden = json.loads((FIXTURES / f"golden_{context_length}.json").read_text())
    assert golden["context_length"] == context_length and golden["final_position"] == context_length - 1
    assert golden["num_layers"] == 32 and golden["vocab_size"] == 128256
    assert golden["model_id"] == provenance["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert golden["dtype"] == "float32"
    assert golden["reference_top1_id"] == golden["reference_top5_ids"][0]
    digest = hashlib.sha256((json.dumps(ids, separators=(",", ":")) + "\n").encode()).hexdigest()
    assert golden["input_ids_sha256"] == digest
    assert all(
        golden["checkpoint_files_sha256"][name] == value
        for name, value in provenance["checkpoint_files_sha256"].items()
    )
    for name, digest in golden["checkpoint_files_sha256"].items():
        assert Path(name).name == name
        assert _sha256(CHECKPOINT / name) == digest, f"Checkpoint file changed: {name}"
    return ids, golden


# Request this fixture inside the test only after the >4K skip and fixture checks.
# Owning the small mesh fixture here makes the early skip independent of fixture order.
@pytest.fixture
def book_mesh_device():
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# Every chunk traverses all 32 real layers. The final valid token passes through
# final RMSNorm and LM head, then its eight TP vocabulary shards form 128256 logits.
# Acceptance means the HF reference's highest-ranked token appears in TT's top five;
# the printed book continuation itself is not used as the expected model prediction.
@pytest.mark.parametrize("context_length", [2048, 4096, 8192, 16384, 32768, 65536])
def test_prefill_book_top5(context_length, request, record_property):
    if context_length > 4096:
        pytest.skip("Book full-model accuracy beyond 4K is not enabled yet")

    ids, golden = _load_book_golden(context_length)

    import torch

    import ttnn
    from models.demos.llama_3p1_8b_d_p.tt.input import upload_token_chunk
    from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
    from models.demos.llama_3p1_8b_d_p.tt.model import PrefillModel

    mesh_device = request.getfixturevalue("book_mesh_device")
    expect_error = request.getfixturevalue("expect_error")
    model = PrefillModel(mesh_device, CHECKPOINT, num_layers=32, max_seq_len=context_length)
    cache = None
    try:
        assert model.num_layers == len(model.layers) == 32
        assert [layer.layer_idx for layer in model.layers] == list(range(32))
        assert model.head is not None
        cache = allocate_kv_cache(mesh_device, model.mesh_config, max_seq_len=context_length)
        ttnn.synchronize_device(mesh_device)
        started = time.perf_counter()
        for start in range(0, context_length, 1024):
            end = min(start + 1024, context_length)
            tokens = upload_token_chunk(
                mesh_device,
                ids[start:end],
                actual_start=start,
                actual_end=end,
                max_seq_len=context_length,
            )
            output = None
            try:
                output = model.prefill_chunk(
                    tokens,
                    cache,
                    slot_idx=0,
                    actual_start=start,
                    actual_end=end,
                    skip_lm_head=end != context_length,
                )
                if start == 0:
                    # A later chunk must not skip even one cache page. Rejection must leave the
                    # valid prefix usable; the next normal chunk and final top-5 check still run.
                    with expect_error(ValueError, "beyond populated cache prefix 1024"):
                        model.prefill_chunk(
                            tokens,
                            cache,
                            slot_idx=0,
                            actual_start=1056,
                            actual_end=1088,
                        )
                if end == context_length:
                    assert tuple(output.shape) == (1, 1, 256, 16032)
                    last_position = context_length - 1
                    sp = (last_position // 256) % 4
                    row = (last_position - start) % 256
                    shards = ttnn.get_device_tensors(output)
                    assert len(shards) == 32
                    logits = torch.cat([ttnn.to_torch(shards[sp * 8 + tp])[0, 0, row].float() for tp in range(8)])
            finally:
                if output is not None:
                    output.deallocate(True)
                tokens.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        assert tuple(logits.shape) == (128256,) and torch.isfinite(logits).all()
        top5 = logits.topk(5).indices.tolist()
        forward_and_readback_seconds = time.perf_counter() - started
        # Decode only after device readback; tokenizer setup is outside forward timing.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        result = {
            "context_length": context_length,
            "final_position": context_length - 1,
            "num_layers": 32,
            "vocab_size": logits.numel(),
            "reference_top1_id": golden["reference_top1_id"],
            "reference_top1_token": tokenizer.decode(
                [golden["reference_top1_id"]],
                clean_up_tokenization_spaces=False,
            ),
            "tt_top5_ids": top5,
            "tt_top5_tokens": [tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in top5],
            "forward_and_readback_seconds": forward_and_readback_seconds,
        }
        record_property("book_top5", json.dumps(result))
        print(json.dumps(result), flush=True)
        assert golden["reference_top1_id"] in top5, result
    finally:
        if cache is not None:
            cache.k.deallocate(True)
            cache.v.deallocate(True)
        model.close()
