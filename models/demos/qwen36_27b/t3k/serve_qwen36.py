# SPDX-License-Identifier: Apache-2.0
"""Minimal vLLM OpenAI-API launcher for Qwen3.6-27B on TT (T3K, TP=8).

Registers the transformers config shim (qwen3_5 arch) and the TT model, then runs
vLLM's OpenAI API server. Mirrors tt-inference-server's run_vllm_api_server.py but
without the ModelSpec-JSON harness. Flags follow the OpenClaw/coder_next template.

Run inside the tt-inference-server dev image (vllm 628d4dc + torch 2.7.1) with OUR
tt-metal mounted at /home/yito/tt-metal (so the _ttnn.so symlink resolves), e.g.:
  TT_METAL_HOME, LD_LIBRARY_PATH (build_Release dirs), PYTHONPATH (our tt-metal),
  MESH_DEVICE=T3K, VLLM_RPC_TIMEOUT=1800000.
"""
import os
import runpy
import sys

# 1) Make transformers recognize the qwen3_5 / qwen3_5_text config (config-only shim).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_config_shim  # noqa: F401  (registers AutoConfig on import)

# 2) Register the TT model. vLLM's TT platform prepends "TT" to the HF architecture
#    name (Qwen3_5ForConditionalGeneration -> TTQwen3_5ForConditionalGeneration).
from vllm import ModelRegistry

ModelRegistry.register_model(
    "TTQwen3_5ForConditionalGeneration",
    "models.demos.qwen36_27b.tt.generator_vllm:Qwen3_5ForConditionalGeneration",
)

# 3) Launch the OpenAI API server. The launch (argv + runpy) MUST be under the
#    __main__ guard: vLLM spawns the engine process, which re-imports this module;
#    the module-level shim+register above run in the child (needed), but the server
#    launch must NOT recurse (else multiprocessing "start before bootstrap" error).
def _main():
    MODEL = os.environ.get("QWEN36_MODEL", "/home/yito/work/qwen36_27b_hf")
    default_args = [
        "--model", MODEL,
        "--served-model-name", "qwen3.6-27b",
        "--host", "0.0.0.0", "--port", os.environ.get("SERVICE_PORT", "8000"),
        "--block-size", "64",
        "--max-num-seqs", os.environ.get("QWEN36_MAX_NUM_SEQS", "1"),
        "--max-model-len", os.environ.get("QWEN36_MAX_MODEL_LEN", "1024"),
        "--no-enable-prefix-caching",
    ]
    # TT decode trace is controlled by override-tt-config "trace_mode" (NOT --enforce-eager,
    # which only affects CUDA graphs). Default "none" = eager (correct multi-request,
    # ~440 ms/tok). QWEN36_TRACE=1 -> "all" = captured-trace decode (~156 ms/tok, correct
    # for a single stream; multi-request trace has a known nondeterministic issue).
    trace_mode = "all" if os.environ.get("QWEN36_TRACE") else "none"
    default_args += ["--override-tt-config", '{"enable_model_warmup": false, "trace_mode": "%s"}' % trace_mode]
    extra = os.environ.get("QWEN36_VLLM_ARGS", "")
    sys.argv = ["vllm"] + default_args + (extra.split() if extra else [])
    print(f"[serve] launching vLLM api_server: {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    _main()
