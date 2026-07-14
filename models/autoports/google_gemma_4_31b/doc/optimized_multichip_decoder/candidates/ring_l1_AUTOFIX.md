# AutoFix: ring persistent-scratch coexistence

## Starting evidence

- Source diagnosis: `ring_l1_AUTODEBUG.md`.
- Original ordered ring run: `ring_topology_final_graph.log`.
- Symptom: after traced decode materialized the shared persistent async-CCL
  scratch, full-attention prefill SDPA collided by 2,112 bytes (static CB end
  1,243,904; L1 buffer start 1,241,792).

## Hypothesis experiments

### Reduce full-prefill K chunk

- Hypothesis: Q=128/K=64 would retain one-Q-chunk scheduling and shrink the
  double-buffered K/V SDPA circular buffers enough to coexist with persistent
  decode scratch.
- Experiment: exact ordered full-attention decode-PCC then warmed-latency run
  under ring topology with `GEMMA4_PREFILL_SDPA_KCHUNK=64`.
- Result: decode PCC passed at 0.9998866201335782. The subsequent prefill still
  failed, now earlier in RMS norm: static CB end 1,237,760 versus persistent L1
  buffer start 1,225,408 on cores 0--3, a 12,352-byte overlap.
- Verdict: refuted as a complete fix. The first error was adapted and retried;
  it exposed the broader requirement that all prefill programs coexist with
  retained decode scratch.
- Evidence: `ring_topology_k64_full.log` and
  `ring_topology_k64_full.xml`.

### Spread the MLP-down persistent intermediate

- Hypothesis: retain the stable shared scratch but shard the decode MLP-down
  result over 24 rather than 14 cores. At 14 cores its BF16 TP4 intermediate
  consumes 98,304 bytes/core (`32 * (5376 / 14 * 4) * 2`). At 24 cores it
  consumes 57,344 bytes/core, freeing 40,960 bytes/core, substantially more
  than the observed 12,352-byte deficit. The prior geometry sweep also made 24
  cores the lowest-latency validated larger-core choice.
- Intervention: candidate-only `GEMMA4_MC_MLP_CORES=24` test plumbing preserves
  the selected packed-BFP8 policy and changes only the already-swept geometry
  fields (`decode_num_cores=24`, gate/up and down `in0_block_w=7`).
- Verification: the exact ordered four-test ring command with
  `GEMMA4_PREFILL_SDPA_KCHUNK=64 GEMMA4_MC_MLP_CORES=24` passed. PCC was
  0.9999591238 (sliding) and 0.9998899999 (full); warmed prefill was
  2.437683/2.443858 ms and traced decode was 0.471058/0.521757 ms.
- Verdict: verified as the capacity fix, rejected as the selected topology.
  The selected Linear/14-core default is faster: 2.390943/2.2507595 ms prefill
  and 0.469603/0.5191525 ms traced decode. Ring+K64+24 cores therefore loses
  1.95%/8.58% on prefill and 0.31%/0.50% on decode for sliding/full respectively.
- Evidence: `ring_topology_k64_mlp24.log` and
  `ring_topology_k64_mlp24.xml`.

## Safety boundary

Do not clear or deallocate the persistent pool between prefill and decode.
Trace capture depends on stable buffer addresses, and test-only reclamation
would hide rather than solve full-model phase coexistence. Linear topology and
the default 14-core policy remain unchanged by this candidate experiment.

## Final status

Fixed for the bounded ring candidate and rejected with complete correctness,
coexistence, and warmed-latency evidence. No production-path change is needed:
the faster selected Linear/14-core path remains the default.
