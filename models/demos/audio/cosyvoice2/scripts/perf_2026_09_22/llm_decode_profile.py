"""Item 1 of the 2026-09-22 round: where does the traced LLM decode's 9.8 ms/token go?

Breaks each traced decode step (`TtQwen2LM._decode_step_traced`, see tt/llm/qwen2lm.py)
into four phases and times each with a device sync boundary:

  host_write  -- building the 3 host tensors (token/pos/rope) + copy_host_to_device_tensor x3
  device      -- ttnn.execute_trace (blocking=False) + an explicit sync to isolate kernel time
  readback    -- _logits_to_host: ttnn.to_torch + .float().reshape()[...] slice
  sampling    -- the host-side RAS draw (tt/llm/sampling.py's ras_sampling), timed via a
                 wrapper on the module-level name generate() resolves at call time

This necessarily adds synchronization points the production hot path does not have (it lets
device and host overlap where it can), so two numbers are reported per utterance: the
UNINSTRUMENTED per-token time (the real 9.8 ms/token path, no extra syncs) and the
INSTRUMENTED breakdown (extra syncs added). If the instrumented total is materially higher
than the uninstrumented one, that gap itself is the cost of the overlap this breakdown
destroys -- reported explicitly, not papered over.

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 300 /opt/venv/bin/python llm_decode_profile.py
from a directory other than the repo root. COSYVOICE2_SCRATCH / OUT_DIR as usual.
"""
import json
import os
import sys
import time
import types

import torch
import ttnn

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/home/user/tt-metal/models/demos/audio/cosyvoice2/scripts/vocoder_debug_2026_09_20")

from cv2_frontend import extract_speech_tokens

from models.demos.audio.cosyvoice2.tt.checkpoint import build_local_qwen2_checkpoint_dir, load_checkpoint_file

SCRATCH = os.environ.get("COSYVOICE2_SCRATCH", "/tmp/cosyvoice2_stage1_eval")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cosyvoice2_perf_2026_09_22")
os.makedirs(OUT_DIR, exist_ok=True)

print("=== loading real LLM checkpoint ===")
llm_sd = load_checkpoint_file("llm.pt")
local_dir = build_local_qwen2_checkpoint_dir(llm_sd, f"{SCRATCH}/qwen2_local_ckpt")
import shutil

from huggingface_hub import hf_hub_download

for fn in ["tokenizer_config.json", "vocab.json", "merges.txt"]:
    p = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename=f"CosyVoice-BlankEN/{fn}")
    shutil.copy(p, f"{local_dir}/{fn}")
os.environ["HF_MODEL"] = local_dir

st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")
import onnxruntime as ort

st_session = ort.InferenceSession(st_path, providers=["CPUExecutionProvider"])

print("=== loading real LibriSpeech test-clean sample (same prompt as rtf_warm.py) ===")
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
ref_ex = ds[0]
ref_wav16 = torch.tensor(ref_ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
prompt_text = ref_ex["text"].capitalize() + "."
prompt_tokens = extract_speech_tokens(st_session, ref_wav16)
print(f"prompt_tokens: {prompt_tokens.shape}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(local_dir)
prompt_text_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)

TEXTS = [
    "Please close the door when you leave.",
    "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch.",
    "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays.",
    "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day.",
]

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=50_000_000)


def sync():
    ttnn.synchronize_device(device)


try:
    print("=== building real LLM (TtQwen2LM, use_decode_trace=True) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm import qwen2lm as llm_module
    from models.demos.audio.cosyvoice2.tt.llm import sampling as sampling_module
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)

    # ---------------------------------------------------------------- part A: real, uninstrumented per-token cost
    print("\n=== PART A: uninstrumented per-token cost (the real production path) ===")
    uninstrumented = {}
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        sync()
        t0 = time.perf_counter()
        toks = tt_llm.generate(
            ids,
            prompt_speech_ids=prompt_tokens,
            max_tokens=int(tl * 20),
            min_tokens=int(tl * 2),
            sampler="ras",
            seed=0,
        )
        sync()
        total = time.perf_counter() - t0
        # First decode step includes compile+capture, not steady-state -- exclude it from the
        # per-token average the same way the 9.8 ms/token figure was derived.
        n_steps = max(len(toks) - 1, 1)
        per_tok = total / n_steps
        uninstrumented[ui] = per_tok
        print(f"[utt {ui}] tokens {len(toks):3d}  total {total:6.3f}s  per-token (excl. first) {per_tok*1000:5.2f} ms")

    # ---------------------------------------------------------------- part B: instrumented breakdown
    print("\n=== PART B: instrumented breakdown (extra sync boundaries -- see module docstring) ===")

    timers = {"host_write": 0.0, "device": 0.0, "readback": 0.0, "sampling": 0.0}
    counts = {"steps": 0}

    orig_decode_step_traced = tt_llm._decode_step_traced

    def instrumented_decode_step(self, token, pos):
        if self._trace_id is None:
            # First call of a generate(): compiles + captures. Not steady state; delegate
            # untouched so capture semantics are exactly the production path's.
            return orig_decode_step_traced(token, pos)

        t0 = time.perf_counter()
        tokens_host = ttnn.from_torch(
            torch.tensor([token] + [0] * 31, dtype=torch.int32).reshape(1, 1, 1, 32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pos_host = ttnn.from_torch(torch.tensor([pos]), dtype=ttnn.int32)
        rope_host = self.rope_setup.get_rot_idxs(torch.tensor([pos]), on_host=True)
        ttnn.copy_host_to_device_tensor(tokens_host, self._trace_tokens)
        ttnn.copy_host_to_device_tensor(pos_host, self._trace_pos)
        ttnn.copy_host_to_device_tensor(rope_host, self._trace_rot_idxs)
        sync()
        timers["host_write"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        sync()
        timers["device"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        result = self._logits_to_host(self._trace_logits)
        timers["readback"] += time.perf_counter() - t0

        counts["steps"] += 1
        return result

    tt_llm._decode_step_traced = types.MethodType(instrumented_decode_step, tt_llm)

    orig_ras_sampling = sampling_module.ras_sampling

    def timed_ras_sampling(*a, **k):
        t0 = time.perf_counter()
        r = orig_ras_sampling(*a, **k)
        timers["sampling"] += time.perf_counter() - t0
        return r

    sampling_module.ras_sampling = timed_ras_sampling

    instrumented_totals = {}
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        for k in timers:
            timers[k] = 0.0
        counts["steps"] = 0
        sync()
        t0 = time.perf_counter()
        toks = tt_llm.generate(
            ids,
            prompt_speech_ids=prompt_tokens,
            max_tokens=int(tl * 20),
            min_tokens=int(tl * 2),
            sampler="ras",
            seed=0,
        )
        sync()
        total = time.perf_counter() - t0
        n = counts["steps"] or 1
        breakdown = {k: v / n * 1000 for k, v in timers.items()}
        instrumented_per_tok = sum(breakdown.values())
        instrumented_totals[ui] = instrumented_per_tok
        print(
            f"[utt {ui}] steps {n:3d}  instrumented per-token {instrumented_per_tok:5.2f} ms "
            f"(host_write {breakdown['host_write']:.2f}  device {breakdown['device']:.2f}  "
            f"readback {breakdown['readback']:.2f}  sampling {breakdown['sampling']:.2f})  "
            f"uninstrumented {uninstrumented[ui]*1000:.2f} ms  "
            f"overhead {(instrumented_per_tok - uninstrumented[ui]*1000):+.2f} ms"
        )

    print("\n=== RESULT ===")
    print(
        json.dumps(
            {
                "uninstrumented_ms_per_token": {k: v * 1000 for k, v in uninstrumented.items()},
                "instrumented_ms_per_token": instrumented_totals,
            },
            indent=2,
        )
    )
finally:
    tt_llm.release_decode_trace()
    ttnn.CloseDevice(device)
