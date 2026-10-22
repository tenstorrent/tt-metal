# Decoder startup: incompatible harvested DRAM layout

## Facts and observations

- Initial decoder on b10u02, allocation142549, launched00:59:10.867UTC and exited1 at01:01:24.942UTC, before requests.
- All four ranks failed allocating KV-cache tensors. First error at01:01:04.244: shard core(7,0) has no DRAM bank; device exposes7.
- Stack points to unchanged Blaze attention_tensors.py:695, ttnn.from_torch for k_cache, after kv_cache_dram_shard constructs its memory config.
- Saved initial/post-build tt-smi telemetry shows all32 b10 chips have exactly7 enabled GDDR banks: masks fe,fd,fb,f7,ef,df,bf,7f, four chips each. DRAM-ready=true means enabled banks passed; it does not guarantee8 banks.
- Retained b09 prefill has all32 masksff. Prior successful b08 decoder also has all32 masksff in saved evidence.
- All4 decoder ranks logged device-driver closure. After exit, process and fuser checks found no task decoder descendants or device holders. No reset was used.
- Local controller92598 was verified and stopped01:03:18, preserving independent prefill103710. Publisher remains read-only until replaced.

## Hypotheses and experiments

H1 (high confidence): unchanged eight-bank decoder sharding is incompatible with harvested seven-bank b10 hardware. The explicit allocator error and independent telemetry agree. Proposed resolution: use a verified full-eight-bank individual Galaxy, preserving source, capacity and tensor layout.
H2 (rejected for this failure): bad cached metadata/EBADF from prior experiment. No EBADF; failure is live KV-cache allocation, with the same deterministic bank error on every rank.
H3 (no supporting evidence): new prefill commit causes decoder failure. Decoder uses separate unchanged Metal44d66500 and Blaze d6d0b09f; failing process stack and hardware telemetry explain the error without the new prefill code.

Read-only investigation: compare first traceback to saved masks and prior successful host. Result confirms a hardware/layout incompatibility; no cache or source changes justified.

## Recovery state / unknowns

Replacement b04u02 reserved under142578, ending2026-10-22 08:19:54UTC; verifieddivanovic RUNNING. B10 release requested01:05:29 after no-holder/process check; verify terminalrelease. B04 must pass32device/DRAM health AND all32 masksff before launch. Prefill remains loading on b09. No new requests have run. Replacement capture success remains unverified. Preserve failed attempt separately and do not claim end-to-end recovery until its owner is ready and measurements pass.

### Recovery checkpoint01:11UTC

B04 health passed32/32 with all32 masksff. B10 terminal CANCELLED verified viaSlurm; no further host access. Failed attempt archived under evidence/failed/decode-attempt1-dram7. Replacement native build110322 launched01:09:42.813374. Sole continuation93538 and publisher93539 launched01:10:15; retained prefill103710, zero cases. New decoder startup has not yet launched. Guard checks passed; current deadlines reduced to fit replacement reservation.

### Recovery checkpoint01:27UTC

Replacement b04 native build and TTNN import passed with exit0 at01:24:28.062541 (885.249s from launch). Fresh prelaunch health01:24:45 again confirmed32devices/32DRAM/all32 masksff. Sole controller93538 launched decoder174718 at01:24:49.738280. All four model ranks175157–175160 were live01:26:59; image capture has begun with no detected rank errors or exit. This passes the initial allocation failure point, but full41-image readiness and P/D requests remain unverified. Retained prefill103710 has been ready since01:12:07.660 and remains live. No source/cache repair or reset was used.

### Later control interruption

At01:36UTC the local coordinator exited on an SSH authentication-stage timeout while the replacement decoder continued capturing. Both models and allocations were retained. See [SSH control investigation](b3b5-ssh-control.md) for evidence, uncertainty and the guarded continuation. This was not a repeat of the DRAM-bank allocator failure.
