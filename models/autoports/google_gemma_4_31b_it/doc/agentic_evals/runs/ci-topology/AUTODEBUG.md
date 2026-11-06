# Terminal CI startup: physical topology cannot satisfy strict fabric mapping

This isolated investigation inspected collected logs and source only. No device commands, model changes, or reset were performed.

Observed contract failure: both initial startup and its retry die in stock plugin worker.set_fabric -> ttnn.set_fabric_config -> topology_mapper.cpp620, before autoport initialization and before Terminal task execution. CI logs report an auto-discovered four-ASIC 2x2_Mesh_flat grouping, with a physical D2-D3 edge offering one channel where the target requires two. Relaxed discovery can find a placement and warns about that deficit. Strict fabric mapping then exhausts solutions and reports target node0 overconstrained.

Source checks:
- Stock plugin utils/dp_discovery.py maps P300x2 to (1,4), so MESH_DEVICE=P300x2 matches the validated local configuration. A 2x2 grouping inside fabric discovery is not proof the requested model mesh was2x2.
- worker.py:1005-1022 applies FABRIC_1D_RING and STRICT_INIT as measured locally. open_mesh_device applies fabric before the actual model mesh opens.
- tt_metal/fabric/mesh_graph_descriptor.cpp:258-261 creates strict channel policy for ordinary reliability modes. topology_mapper.cpp:552-554 imports the descriptor policy into the solver.
- The current autoport uses num_links=2 in multichip_decoder.py and model.py. Silently reducing fabric channel requirements would not by itself validate these collective calls.
- Local stage11 configuration-b8.json shows the same P300x2 (1,4), strict ring, auto-discovery and unknown B850M-C motherboard warning, but successful startup. That motherboard warning alone is not a causal diagnosis.

Conclusion: failed CI physical/discovered connectivity is inconsistent with the strict two-channel target. Logs do not establish whether the missing link is transient hardware health, cabling, or an auto-discovery defect. A model-registration shape fix is unsupported. Do not cancel another machine's run merely because this runner failed.

Minimal next action: infrastructure retry through the unchanged granted stock route on an available same-type runner, preserving immutable source configuration and failed evidence. The registered native parser fix may independently accompany the retry. If a repeat fails on the same physical node, capture system-health/topology evidence and apply tt-device-usage recovery rather than weaken channel requirements. A local mesh smoke cannot distinguish the failed remote runner's channel deficit; repeated healthy local serving already proves this launch configuration can open.

Host-first discrimination completed without device time: P300x2 alias parity, launch-config parity, explicit deficient-edge evidence, and strict-vs-relaxed mapping path. No source patch is proposed; live retry remains required.
