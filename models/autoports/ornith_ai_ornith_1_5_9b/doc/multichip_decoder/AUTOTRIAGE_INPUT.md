# Geometry hang investigation context

The stalled process imported the original multichip implementation, retained in
initial_multichip_decoder.py in this directory. Production was updated afterward
to incorporate separately validated collective candidates; this has no bearing
on the live isolated matmul (it calls ttnn.linear directly).

Invocation: timeout 1800 python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_geometry --layer 0
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/geometry_0.json
stdout/stderr geometry_0.log. PID2111329, timeout PID2111328. Started ~19:24:51 UTC.
Do not terminate; existing timeout ends it. Parent captured triage at ~19:26 UTC.

geometry_0.json contains 132 completed cases, ending with down projection
cores32/block1/readers4. The next iteration is cores32/block1/readers8.
Triage identifies MatmulDeviceOperation309, BF16 input [1,1,1,3072] width-sharded L1,
BFP8 weights [3072,4096] width-sharded DRAM, four identical shapes on TP4 mesh.
No CCL runs in the swept op. All previous real-weight candidate PCCs passed.

Parent constraints: user allows changes only in model multichip_decoder.py,
its tests and docs. Shared-kernel changes are outside this stage authorization.
Report exact model-local padding/sharding/config remedy or proven unsupported
geometry, with smallest discriminating experiment. Preserve evidence.


Correction after isolated AutoTriage: the last completed JSON candidate was down
cores=32, block=3, readers=8. The hanging next candidate was down K3072,
requested input/output cores=64, block=1, readers=1. See AUTOTRIAGE.md and
geometry_0_before_repair.json. The initial 32-core/eight-reader suspicion above
was superseded by operation 309's compiled runtime arguments and coordinates.
