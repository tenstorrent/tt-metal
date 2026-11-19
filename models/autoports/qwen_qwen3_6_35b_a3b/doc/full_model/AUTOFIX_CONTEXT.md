# Maximum-context L1 lifetime repair

Focused repair and all40 maximum-context acceptance **PASS**. Final-source all40 prefill and teacher accuracy refresh also **PASS** (top1=98/100, top5=100/100, top100=100/100 each). CPU documentation/source comparison only; completed acceptance artifacts were inspected after the parent confirmed exit0. No device operation or acceptance rerun was performed for this update.

The original all40 run failed during short bounded-prefill preparation, before any long-prefill/decode acceptance call. Singleton SDPA's static CB region ended at920640 while the lowest allocated L1 buffer began at898496, a22144-byte clash. The8192-entry INT32 page table at context262144 requires32768 bytes of SDPA CB9, independent of the small current position. This establishes an allocation-lifetime problem at the advertised context, not a physical DRAM capacity limit. See `AUTODEBUG_CONTEXT.md`, `context_acceptance/result.json`, and `logs/context_acceptance.log:591`.

## Change and preserved contract

`tt/multichip_decoder.py:1735-1738` now deletes `k`, `v`, and the surviving loop alias `src` after both `paged_update_cache` operations have been enqueued, before Q's final memory conversion and SDPA. Python previously retained these dead source heads until `_full_attention` returned. Each source occupies one BF16[32,256] L1 shard: nominal combined footprint32768 bytes. Including `src` is necessary because the completed loop still references V.

The repair is exactly a lifetime shortening plus its explanatory comment. It adds no operation, copy, synchronization, host conversion, precision change, allocator mode, or core placement. The same serialized command queue orders cache writes before subsequent consumers. Q, gate, persistent KV and learned norm-gamma allocations remain owned. SDPA remains8x8 with q_chunk32/k_chunk256, the existing BF16 KV split and compute policy, TP4/EP4 distribution, and replicated BF16 interlayer residual. No advertised context reduction was made.

## Focused device evidence supplied by the parent

| Run | Pressure/configuration | Observed result |
| --- | --- | --- |
| Baseline `l1_repro_before` | Real layers0,3; context262144; resident L1 target571456 bytes | Reproduced SDPA clash: lowest L1 address912064, CB end920640. |
| First repaired `l1_repair` | Same reduced model; stronger resident target585024 | Warmup progressed, but positive-control assertion failed because trace allocation tracking environment variables were unset. This is an invalid tracker-validation launch, not a new model failure. |
| Repaired `l1_repair_tracked` | Real layers0,3; context262144; resident target585024; tracking1, skip-program-cache0 | `FULL_MODEL_PROBE_OK`; positive control rejected an intentionally unsafe live allocation; traced generation returned expected tokens. |

The stronger target adds13568 bytes, matching the observed all40-versus-reduced low-address difference (912064−898496). It is a conservative pressure fixture, not an assertion that reduced execution reproduces every full-stack lifetime. The repaired census records96320 bytes before ballast,488704 bytes of retained ballast, and585024 bytes resident after ballast and generator setup. The actual probe uses one setup-only FP32 ROW_MAJOR one-core allocation to impose that footprint; it does not instantiate29 additional decoder layers.

Tracked result: tokens `[28433,150929,11927,66275]`, model replays3, sampling replays4, captures2, releases2, readbacks4. Its token/position refresh counts6/9 reflect this setup/generate diagnostic; they are not a zero-refresh token-out performance claim. Positive-control error explicitly identifies one live buffer that would be corrupted on trace replay. Log reports7352/7352 JIT hits and normal device close. Parent wall-time observation was approximately33 seconds; log timestamps span14:38:16.371–14:38:47.842 UTC.

Artifacts, relative to this directory:

- `context_acceptance/l1_repro_before.json.provenance.json`, `logs/context_l1_repro_before.log` (failing baseline; no successful result JSON).
- `context_acceptance/l1_repair.json.provenance.json`, `logs/context_l1_repair.log` (tracking unset; no successful result JSON).
- `context_acceptance/l1_repair_tracked.json`, `context_acceptance/l1_repair_tracked.json.provenance.json`, `logs/context_l1_repair_tracked.log` (focused passing evidence).

The tracked invocation recorded by provenance is:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 1200 python models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_probe.py \
  --context 262144 --resident-l1-bytes 585024 --generate --positive-control \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/context_acceptance/l1_repair_tracked.json
```

The Python argv and tracking environment are run-recorded; activation and timeout above provide a bounded reproduction wrapper, not a recovered shell transcript. Preserve these historical artifacts; use a new output basename for any rerun.

## Source provenance comparison

CPU inspection confirmed the current probe/model/generator hashes exactly match all three run sidecars:

| File | SHA256 |
| --- | --- |
| `tests/run_full_model_probe.py` | `ee830e2842b88903751b4c00b431de72b258ee6118778dfdd8a54581492ed05c` |
| `tt/model.py` | `a0e5eb31075ea40326d4be423e907aca216f43abb25ff86c80b402ccf93e1317` |
| `tt/generator.py` | `268f4c3ac0624f3f29e8b680eeb6382a43c01b0f7cadbb20df88ba37823ebeaa` |

The sidecars omit `tt/multichip_decoder.py`, so they do not independently pin the changed helper. Its **inspection-time**, repaired-source SHA256 is `d08412bdbd7bfbb2bcdef2974b1c3f5b924c8a7ffca4590be80316997cf0bf1b`. Removing only the lifetime comment and `del k, v, src` from those bytes yields `67431e27876833c44bf69395a814dae3788c2317206866c74c8182047b088a66`; this reconstructed hash is not a run-recorded baseline hash. Parent identified the before/after launches. `tt/fused_decoder.py` inspection hash is `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`. Final evidence should record the decoder hash alongside model/generator hashes.

## Completed maximum-context acceptance and final-source checks

Completed `context_acceptance/final.json` reportsPASS, one262143-token nonaligned prefill and one decode at position262143. Prefill took5883.448102414011 seconds; decode took0.033730191993527114 seconds. The parent observed exit0 after approximately6277 seconds total. Final position262144, finite logits,240 linear-state chip checks, all10 KV layers×4 chips last-page checks, unchanged preceding cache rows and resident token feedback all pass. Decode replay counters are one model/one sampler with zero token/position/page/activity refresh, readback or capture. The input/output sampled tokens are82/3. Cleanup releases0→2 with no cleanup errors and normal close in the completed log.

The prepared/post-prefill/post-decode censuses all retain25372641280 allocated and8606073856 free bytes per chip, capacity33978715136. This validates endpoint capacity at the unchanged262144 context; it is not a transient peak measurement. The raw inventory remains partial because its run-recorded visitor skipped callable decoder/sampler objects; the allocator census is authoritative. `doc/context_contract.json` has been updated with these exact scopes.

Completed result SHA256: `858e1ad664ca2c04d519e50a30d05119433c79fb7f615e5b8ede86e5e0a0fd4a`. Completed `logs/context_acceptance_lifetime_fixed.log` SHA256: `43b1a2c843001f9c1525a25199f3c0f314d0a1d5afb2fef2c6f0f86b0a74921a`. All inference hashes still match run provenance, including repaired decoder `d08412bdbd7bfbb2bcdef2974b1c3f5b924c8a7ffca4590be80316997cf0bf1b`. The parent subsequently removed only the inventory visitor's `callable(value)` exclusion and added its explanatory comment; reversing that hunk reproduces run-recorded runner SHA `e8aa33c328575e9e2ad3f694263303f7c6c13e10b821f7eb6551896196203585` from current reporting-only SHA `4eec1281be4349698893c9a98316ea4bc5343528040ee9f2bd8017885cc33ebd`. Raw result and historical failures were preserved, with no second long-context acceptance.

The completed current-source all40 AIME reference gates are recorded below; both sidecars include `tt/multichip_decoder.py` and match current production files. These are executed evidence commands, not a request for another run:

```bash
timeout 1200 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_readiness \
  --mode prefill --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/prefill_accuracy_lifetime_fixed.json
timeout 1200 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_readiness \
  --mode teacher --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/teacher_accuracy_lifetime_fixed.json
```

Both retain the exact pinned100-token AIME24 chat-template reference and report top1=98/100, top5=100/100 and top100=100/100; required top5≥98%, top100=100% pass. Teacher forcing exercises the modified singleton decode for all40 layers. Earlier accuracy results remain historical. The current-source reduced layers0/3 watcher replay also passes (`watcher/summary.json`), covering nonaligned/mixed/page/state/sampling cases; it does not claim an all40 or maximum-context watcher run. The installed stage06 context subcheck passes (`logs/stage06_final_gate.log`), while the overall invocation exited2 on serving-contract findings (four resumed-prefill gaps and82 subsequently corrected evidence/declaration formatting findings). No gate rerun or overall pass is claimed. Resumed-prefill, final representative performance and independent review remain separate; this L1/capacity/accuracy repair record waives none of them.
