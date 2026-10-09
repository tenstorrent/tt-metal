# Chunked-prefill model diagnostic, queued Oct 9 2026 UTC

**Hardware is unrun.** This prepares the next correctness check for shorter
scheduler prefill quanta; it does not enable a model capability or change the
current serving policy. The plugin ownership prerequisite is separately
[reproduced and fixed](../chunked-prefill-state-v1/README.md).

The full 64-layer TP4 adapter runs three teacher-forced requests independently,
then with interleaved prefills/decode and explicit physical state permutations.
Both arms use identical chunk boundaries. Eleven full-vocabulary logit outputs
are compared, covering an unaligned continuation, a new request preceding a
continuation, reordered rows and continuation after a decode gather. Attention
pages remain request-owned. Resident decode buckets are enabled. The declared
diagnostic criteria are PCC >= 0.999, relative RMS <= 0.02 and greedy agreement
at all eleven positions. These are short scheduling-equivalence checks, not
GPQA, long-horizon GDN fidelity or performance measurements.

Host logits are used on both arms, matching the plugin's intermediate-prefill
requirement. Actual scheduler integration, final device sampling/RNG continuity,
larger prompts and matching end-to-end qualification remain separate gates.
This experiment does not substitute for them or claim that chunking raises
aggregate throughput. Smaller chunks may reduce decode pauses but add program
launches, trace transitions and reduce prefill batch efficiency.

Validation before launch:

- Twelve new CPU scenario/controller tests pass locally. Deliberately losing a
  continuation slot or omitting a physical gather causes the two schedules to
  disagree; the harness observes actual simulated request histories.
- Native-host preflight: **27 passed, six subtests passed, one intentionally
  skipped hardware test**. Existing adapter prefill and seed tests are included.
- Runtime imports, the original native G0 source/precision binding and the
  disabled chunked-prefill capability were checked without opening hardware.

Persistent unit: `qwen38-chunked-state-v1-20261009.service`, initially PID
`1290750`, invocation `f20791c8f538464a8903cab2712c7ce7`. It waits on the exact
HF-layer unit invocation `aa7a2d34fdd74d538ba2725b99d43311`, including actual
process exit and a completed receipt. This orders it after the native sweep,
head-control G0/GPQA, CPU/image work and optional HF-layer diagnosis. An explicit
no-hardware skip after passing head GPQA is accepted; unproven hardware cleanup
or a predecessor failure is rejected. The shared device lock remains required.

The unit survives SSH/session disconnects. It has a 24-hour lifetime, 22-hour
predecessor wait, a 3600-second pytest bound and 4200-second outer run bound,
160 GiB memory and eight CPU cores. It does not resume after reboot.
Frozen source manifest SHA256:
`49d8a9ee37c9633a98960a3e8cf0268232e5c40630c8ec149792be56318f03fe`.

The first local staging attempt was denied by the SSH sandbox before connecting;
the approved retry passed preflight and launched once. Existing immutable
running sources and image/plugin pins were unchanged.

[Launch command](launch.json), [initial wait receipt](queue-at-launch.json),
[source manifest](source-manifest.json), [CPU JUnit](unit.xml.gz),
[preflight log](preflight.log.gz), [source checks](source-preflight.log.gz).

Remote result directory:
`/home/ttuser/qwen38-artifacts-20261007/chunked-state-v1`.
