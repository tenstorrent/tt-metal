# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-32B topk40k accuracy check (harness, no model change).

Teacher-forced next-token agreement over models/tt_transformers/tests/reference_outputs/Qwen3-32B.refpt
(reference_tokens [1, 1024], top5_tokens [1023, 5] from the CPU reference model), with the 1024-token
sequence padded to 40960 so the long-context prefill path runs: ring SDPA and the >= 16k configs.

Device side: the same model and generator path as the demo's prefill-perf-40k-b1 case (create_tt_qwen_model,
64 layers, bf8 weights, paged KV cache with block 64, Generator.prefill_forward_text with its warmup sweep,
untraced single-user prefill at prefill_seq_len 40960). The 1024 reference tokens are followed by 39936 zero
tokens (the generator's own pad id) and submitted with prompt_len 40960, so prefill_seq_len is 40960.
Attention is causal, so positions 0..1022 do not see the padding.

The pre-norm hidden states the model returns are read at process_output_prefill (where the generator applies
the final norm and LM head to its last token). The final RMSNorm and the LM head are then applied on the host
in fp32 to positions 0..1022, as in test_qwen_model_prefill_ttt.py.

Metrics, printed as one TOPK40K_RESULT line (thresholds live in the tt-ops gate, not here):
  top1 / top5        demo definition (text_qwen_demo.py TokenAccuracy.compute_accuracy): device argmax ==
                     top5_tokens[p, 0], and device argmax in top5_tokens[p, :]. Over all 1023 positions.
  top1_h2 / top5_h2  the same, positions 511..1022 only (the demo's ci-token-matching scored range).
  gt_top1 / gt_top5  vs the text: device argmax == reference_tokens[p + 1], and reference_tokens[p + 1] in
                     the device top-5. ref_gt_top1 / ref_gt_top5: the reference model's own values.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_qwen_demo import create_tt_qwen_model
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import DECODE_FABRIC_CONFIG
from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

REFPT = "models/tt_transformers/tests/reference_outputs/Qwen3-32B.refpt"
PADDED_LEN = 40960
SPLIT = 511  # demo TokenAccuracy: split_point 512 -> first scored prediction is top5_tokens[511]


def _hf_tensor(name):
    ckpt = os.getenv("HF_MODEL")
    if not ckpt:
        pytest.skip("HF_MODEL is not set (Qwen3-32B repo id or checkpoint directory)")
    return load_hf_state_dict_filtered(ckpt, [name])[name]  # local dir or HF repo id


def topk_metrics(dev_logits, ref_top5, ref_tokens):
    """dev_logits [P, vocab]; ref_top5 [P, 5] (reference top-5 ids at position p); ref_tokens [P + 1]."""
    dev_top5 = torch.topk(dev_logits, k=5, dim=-1).indices
    dev_argmax = dev_top5[:, 0]
    nxt = ref_tokens[1:]
    pct = lambda b: b.double().mean().item() * 100
    d1 = dev_argmax == ref_top5[:, 0]
    d5 = (ref_top5 == dev_argmax[:, None]).any(-1)
    return {
        "n": int(dev_logits.shape[0]),
        "top1": pct(d1),
        "top5": pct(d5),
        "top1_h2": pct(d1[SPLIT:]),
        "top5_h2": pct(d5[SPLIT:]),
        "gt_top1": pct(dev_argmax == nxt),
        "gt_top5": pct((dev_top5 == nxt[:, None]).any(-1)),
        "ref_gt_top1": pct(ref_top5[:, 0] == nxt),
        "ref_gt_top5": pct((ref_top5 == nxt[:, None]).any(-1)),
        "dev_argmax": dev_argmax,
    }


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            TRACE_MODEL_KEY_PARAM: "llama3.3-70b-galaxy-qwen",
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "fabric_config": DECODE_FABRIC_CONFIG,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_qwen_topk40k(mesh_device, reset_seeds):
    ref = torch.load(REFPT)
    ref_tokens = ref["reference_tokens"][0].long()  # [1024]
    ref_top5 = ref["top5_tokens"].long()  # [1023, 5]
    n_tok = ref_tokens.shape[0]
    assert ref_top5.shape == (n_tok - 1, 5), ref_top5.shape

    page_params = {"page_block_size": 64, "page_max_num_blocks": 4096}  # as prefill-perf-40k-b1
    model_args, model, page_table, tt_kv_cache = create_tt_qwen_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=PADDED_LEN,
        num_layers=64,
        dummy_weights=False,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    tokens = torch.zeros(1, PADDED_LEN, dtype=torch.long)
    tokens[0, :n_tok] = ref_tokens

    # Read row 0 of the pre-norm output (cols 0..3 hold dim/4 each) for the real run, not the warmup sweep.
    captured = {}
    process_output_prefill = model.process_output_prefill

    def capture(tt_out, last_token_idx, **kwargs):
        if not generator.warming_up_prefill:
            row0 = ttnn.get_device_tensors(tt_out)[: model_args.cluster_shape[1]]
            parts = [ttnn.to_torch(t)[0, 0, : n_tok - 1, :].float() for t in row0]
            captured["hidden"] = torch.cat(parts, dim=-1)[:, : model_args.dim]
            captured["padded_len"] = int(tt_out.shape[2])
            captured["last_token_idx"] = last_token_idx
        return process_output_prefill(tt_out, last_token_idx, **kwargs)

    model.process_output_prefill = capture
    tt_out_logits_all_users = torch.zeros(1, 1, model_args.padded_vocab_size)
    generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[PADDED_LEN],
        enable_trace=True,
        tt_out_logits_all_users=tt_out_logits_all_users,
        sampling_params=SamplingParams(temperature=0, top_k=32, top_p=0.08),
    )
    model.process_output_prefill = process_output_prefill
    assert captured, "process_output_prefill was not called for the reference prompt"
    logger.info(
        f"TOPK40K real_tokens={n_tok} padded_prefill_seq_len={captured['padded_len']} "
        f"last_token_idx={captured['last_token_idx']}"
    )
    assert captured["padded_len"] == PADDED_LEN, captured["padded_len"]

    # Host final norm + LM head in fp32 (as test_qwen_model_prefill_ttt.py).
    norm_w = _hf_tensor("model.norm.weight").float()
    lm_w = _hf_tensor("lm_head.weight").float()  # [vocab, dim]
    x = captured["hidden"]
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + model_args.norm_eps) * norm_w
    dev_logits = (x @ lm_w.T)[:, : model_args.vocab_size]

    m = topk_metrics(dev_logits, ref_top5, ref_tokens)
    out = os.environ.get("QWEN_TOPK40K_OUT")
    if out:
        torch.save({"dev_argmax": m["dev_argmax"], "dev_top5": torch.topk(dev_logits, 5, -1).indices}, out)
    print(
        "TOPK40K_RESULT n=%d top1=%.4f top5=%.4f top1_h2=%.4f top5_h2=%.4f gt_top1=%.4f gt_top5=%.4f "
        "ref_gt_top1=%.4f ref_gt_top5=%.4f"
        % (
            m["n"],
            m["top1"],
            m["top5"],
            m["top1_h2"],
            m["top5_h2"],
            m["gt_top1"],
            m["gt_top5"],
            m["ref_gt_top1"],
            m["ref_gt_top5"],
        ),
        flush=True,
    )
    model.tt_ccl.close()
