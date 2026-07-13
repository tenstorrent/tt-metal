import gc
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.tt.common import create_tt_model

# Match the pytest / demo run contract (export these in your shell too).
os.environ.setdefault("ARCH_NAME", "blackhole")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_default_snapshot = Path(os.environ["HF_HOME"]) / "hub/models--google--gemma-4-31B-it/snapshots/main"
os.environ.setdefault("HF_MODEL", str(_default_snapshot))
os.environ.setdefault(
    "TT_CACHE_PATH",
    str(Path(os.environ["HF_HOME"]) / "tt_cache/google--gemma-4-31B-it"),
)

model_path = os.environ["HF_MODEL"]
cache_path = os.environ["TT_CACHE_PATH"]

if not Path(model_path).is_dir():
    raise SystemExit(
        f"HF_MODEL path does not exist: {model_path}\n"
        "Set HF_MODEL to the snapshot directory, e.g.\n"
        f"  export HF_MODEL={_default_snapshot}"
    )


def timed(label, fn):
    t0 = time.time()
    print(f"[START] {label}", flush=True)
    fn()
    print(f"[DONE]  {label} in {time.time() - t0:.1f}s", flush=True)


def open_mesh_1x4():
    """Match pytest mesh_device fixture: set fabric BEFORE open_mesh_device."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    return mesh


def close_mesh(mesh):
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


mesh = open_mesh_1x4()
print(f"HF_MODEL={model_path}", flush=True)
print(f"TT_CACHE_PATH={cache_path}", flush=True)
try:
    # A) TT-only path (like demo) — should be fast with warm cache
    timed(
        "create_tt_model (no HF)",
        lambda: create_tt_model(mesh, max_batch_size=1, max_seq_len=128, model_path=model_path, create_kv_cache=True),
    )

    # B) HF-first path (like test_full_model) — repro candidate
    from transformers import AutoModelForCausalLM

    def load_hf():
        global hf
        hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    hf = None
    timed("load HF full model", load_hf)
    timed("HF forward 32 tokens", lambda: hf(torch.zeros(1, 32, dtype=torch.long)))
    del hf
    gc.collect()
    timed(
        "create_tt_model after HF",
        lambda: create_tt_model(mesh, max_batch_size=1, max_seq_len=128, model_path=model_path, create_kv_cache=True),
    )
finally:
    close_mesh(mesh)
