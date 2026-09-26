# Independent sparse gate/up K-block search

Actual 4096/128 B1 inputs, final mixed BFP8/BFP4 LoFi policy, down K22 held fixed. Timing is whole-layer warmed traced host time, not device time. Both kinds prefer packed gate K44 on 44 cores; K88, 22-core K44, and legal separate gate/up K44 lose. Exact commands: `last_candidate_commands.json`. All cases preserve the same PCC; this is a program-config change. Defaults will be updated and revalidated.

| Candidate | Minimum decode PCC | Median traced host us |
|---|---:|---:|
| actual_gate_k22_layer0.json | 0.995341957 | 1073.880 |
| actual_gate_k22_layer5.json | 0.995214121 | 1170.943 |
| actual_gate_k44_cores22_layer0.json | 0.995341957 | 1089.837 |
| actual_gate_k44_cores22_layer5.json | 0.995214121 | 1187.749 |
| actual_gate_k44_layer0.json | 0.995341957 | 1070.799 |
| actual_gate_k44_layer5.json | 0.995214121 | 1168.433 |
| actual_gate_k44_separate_layer0.json | 0.995341957 | 1076.930 |
| actual_gate_k44_separate_layer5.json | 0.995214121 | 1177.552 |
| actual_gate_k88_layer0.json | 0.995341957 | 1075.709 |
| actual_gate_k88_layer5.json | 0.995214121 | 1171.281 |
