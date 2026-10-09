# Targeted GDN reader/compute/writer Tracy experiment

The existing BFP8 profiles have no populated NoC/DRAM-utilization or CB-wait
columns. Added diagnostic source annotations around reader CB reservation,
DRAM request/completion, L1 row preparation, compute input waits and math
phases, writer state waits and output completion. Production kernel files
remain unchanged; the isolated test passes annotated strings to generic_op.

B32/B16 each compare one and two input buffers, with original/instrumented
controls, three updates each: 24 kernel calls total. Every rank must pass the
FP32 reference and instrumented state/output must match controls bit-for-bit.
The collector requires all ten zone labels in raw device CSV plus passing
JUnit and clean hardware closure. Raw times include instrumentation overhead;
wait-inclusive regions can overlap, and a read wait alone does not prove NoC
congestion. NoC routing or reader placement needs a subsequent controlled A/B
if these timings indicate a transfer bottleneck.

v1 stopped during CPU preflight: the token-preservation test's marker regex
omitted digits and failed to remove the L1 label. Fixed the regex, verified
exact token preservation and kept the failed preflight. No v1 hardware ran.
v2 passed472 CPU tests,40 subtests, one unrelated skip and hardware collection.

Persistent unit `qwen38-gdn-phase-profile-v2-20261009.service`, invocation
`fac6b2aead2046e0a737ad04d6d6d0ae`, PID3016459 was active on launch. It waits for
`/tmp/tt-device.lock`, with2h/48GiB/8CPU limits,90-minute capture deadline and
20-minute pytest limit. Artifact monitoring caps total output at4GiB, each file
at1GiB and requires16GiB free. It survives disconnect, not reboot. Launch and
preflight are not hardware results. No runtime source, firmware or NFS changes.
