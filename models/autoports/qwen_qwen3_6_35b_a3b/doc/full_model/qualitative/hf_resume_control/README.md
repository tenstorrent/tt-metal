# HF resumed-prefill numerical control

The original pinned HF model reproduces all three case-level PCC gate failures.
The TT packaged result remains **FAIL**. This control does not change thresholds,
claim a passing TT resume gate, or establish generated-text quality.

Exact bounded command, from the configured checkout:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
timeout 1800 python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/run_control.py \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/run.log 2>&1
```

The actual run exited0 in25.45s. Model/tokenizer revision:
`345d467b117d1fd5eda9b879a5988de9ae5f47a7`. The original
`AutoModelForCausalLM.from_pretrained(..., revision=..., local_files_only=True)`
uses BF16, CPU8threads and its default attention/cache implementations, without
model-policy overrides. Transformers automatically uses its Torch recurrent
implementation because optional fast-path libraries are absent; this is recorded
in `run.log`. No TTNN import or TT device access occurred.

| Length/start | HF PCC mean/min | TT PCC mean/min | HF/TT top1 agreement | HF worst position |
|---|---|---|---|---:|
| 257/128 | .987027/.496343 | .993512/.858570 | 64/65,64/65 | 135 |
| 1025/260 | .991617/.952856 | .993122/.968665 | 64/65,65/65 | 300 |
| 1025/1024 | .988465/.988465 | .987021/.987021 | 1/1,1/1 | 1024 |

Every final-position top1 matches. Thresholds remain meanPCC>=.99,
minPCC>=.98, top1>=.98 and final-top1 equal. Natural-book tokenization and selected
positions are the installed checker's exact helpers: first64 continuation
positions plus the last position, deduplicated. `prompt_tokens.json` preserves
the first1025 tokenizer IDs. Full and resumed paths use fresh independent
requests; each continuation retains its own actual prefix cache and correct
absolute positions. Only requested hidden rows reach the HF LM head through
its native `logits_to_keep` interface.

`result.json` records every selected HF row's PCC, full/resumed top1 IDs, worst
position, finite checks, cache-length checks, tensor-content hashes and unchanged
Torch RNG state for all8 forward calls. `verdict.json` adds explicit mismatch
positions and the corresponding TT aggregate metrics. The TT artifact does not
contain per-row values: exact overlap of TT/HF outliers is **not established**.
HF reproducing a case-level threshold failure must not be represented as proof
that the same individual TT row is a model limitation.

The numeric rubric was written before loading/running HF. Opaque permuted metric
records were evaluated by the unchanged threshold function, without source labels;
`blind_scores.json` precedes unblinding into `result.json`. Source identities are
preserved separately in `blind_labels.json`. No human content scorer is used.
Seeds0–9 are explicit reused labels for one deterministic measurement per case,
not ten independent trials. All measured HF forwards used seed0 and consumed no
Torch RNG. The existing TT run is also reused once. This is an efficient
numerical control authorized for this deterministic check; it is not the
qualitative skill's ten-seed sampled-content experiment and cannot support a
statistical20-percentage-point failure-rate waiver.

The raw natural continuation format intentionally matches the failing numerical
probe. No chat template or model-card sampling is applied because that would
change the tested token/cache path. Separate correctly rendered HF/TT shared
qualitative controls already exist in the parent directory. No qualitative
failure is diagnosed by this control.
