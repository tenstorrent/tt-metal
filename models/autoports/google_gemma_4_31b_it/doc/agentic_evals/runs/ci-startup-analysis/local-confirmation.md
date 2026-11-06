# Coordinator-only local confirmation

No command below has been run by the investigator. Use one local queue job, B8, C2,0.25device-hour box; do not overlap another owner. This test uses existing files through aliases, not an empty cache, copied weights or cold compilation.

Host setup and identity (no device):

```bash
python_env/bin/python - <<'PY'
from pathlib import Path
root=Path.cwd()/'bringup/artifacts/cache-env-validation'
for alias,target in [(root/'weights/hub',Path('/home/bringup/.cache/huggingface/hub')),(root/'tt-metal-jit',Path.cwd()/'bringup/artifacts/tt-metal-cache')]:
    alias.parent.mkdir(parents=True,exist_ok=True)
    if alias.exists() or alias.is_symlink():
        assert alias.resolve()==target.resolve(), alias
    else:
        alias.symlink_to(target,target_is_directory=True)
PY
HF_HUB_CACHE="$PWD/bringup/artifacts/cache-env-validation/weights/hub" python_env/bin/python - <<'PY'
import hashlib,json,os
from pathlib import Path
from huggingface_hub import hf_hub_download,constants
revision='842da3794eaa0b77d5f08bae87a17459d91ff475'
original=Path('/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots')/revision
assert Path(constants.HF_HUB_CACHE).resolve()==Path('/home/bringup/.cache/huggingface/hub').resolve()
index=Path(hf_hub_download('google/gemma-4-31B-it','model.safetensors.index.json',revision=revision,local_files_only=True))
files=['model.safetensors.index.json',*sorted(set(json.loads(index.read_text())['weight_map'].values()))]
checks=[]
for name in files:
    actual=Path(hf_hub_download('google/gemma-4-31B-it',name,revision=revision,local_files_only=True))
    assert os.path.samefile(actual,original/name),name
    checks.append({'file':name,'same_inode':True,'bytes':actual.stat().st_size})
print(json.dumps({'resolved_cache':constants.HF_HUB_CACHE,'revision':revision,'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'files':checks},indent=2))
PY
```

For the serving launch, use the existing readiness entrypoint directly with the same environment setup as tests/run_serving.sh, then set both aliases **after** setup. Calling run_serving.sh itself would overwrite TT_METAL_CACHE, so would not confirm that candidate variable.

```bash
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.9
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export VLLM_TT_PLUGIN_ROOT=/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09
export TT_MODEL_CLASS_OVERRIDES='TTGemma4ForConditionalGeneration=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM,TTGemma4ForCausalLM=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM'
export HF_HUB_CACHE="$PWD/bringup/artifacts/cache-env-validation/weights/hub"
export TT_METAL_CACHE="$PWD/bringup/artifacts/cache-env-validation/tt-metal-jit"
export XDG_CACHE_HOME="$PWD/bringup/artifacts/xdg-cache"
export VLLM_SERVER_DEV_MODE=1
CACHE_TEST_SNAPSHOT=/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
timeout 900 python_env/bin/python -m readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b_it \
  --hf-model "$CACHE_TEST_SNAPSHOT" --mesh-device P300x2 \
  --max-model-len 262144 --max-num-seqs 8 --block-size 32 \
  --server-timeout 600 --sampling-profile smoke --stages serve \
  --benchmark-prompt-len 4096 --benchmark-output-len 128 \
  --output-dir models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/ci-startup-analysis/local-server \
  --tt-config '{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args "--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --tokenizer-revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --language-model-only --async-scheduling --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --override-generation-config '{\"top_k\":32}' --served-model-name google/gemma-4-31B-it $CACHE_TEST_SNAPSHOT"
```

After health, coordinator sends the existing parser-protocol/paired-terminal-request.json twice concurrently, preserves responses, and applies audit_responses.py --pair against paired-candidate-0.json and paired-candidate-1.json (168tokens each). Verify live API process environment resolves both aliases. Stop idle server through the established owned graceful shutdown. No commands from model responses execute. Local confirmation proves routing and paired output identity only; CI startup savings remain unmeasured.
