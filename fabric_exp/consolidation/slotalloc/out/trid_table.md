Unit checks (assign_transaction_ids; receivers as slots, a = first-level acks):

| case | result | new |
| --- | --- | --- |
| 2D lone 16 | PASS | 4/4@0 (4 IDs) |
| 1D 32 | PASS | 4/8@0 (8 IDs) |
| ring 32a: 32 -> 16 (halve a power of 2) | PASS | 1/16@0! (16 IDs) |
| ring 20a: 20 -> 16 (round down, not 8) | PASS | 1/16@0! (16 IDs) |
| ring 24a: 24 -> 16 | PASS | 1/16@0! (16 IDs) |
| torus 16a + VC1 4: 16 -> 8 | PASS | 1/8@0! | 1/4@8 (12 IDs) |
| 20a + 4: 20 -> 16 -> 8 | PASS | 1/8@0! | 1/4@8 (12 IDs) |
| 12a + 8a: 12 -> 8 | PASS | 1/8@0! | 1/8@8 (16 IDs) |
| three VCs 16 / 6 / 2: fits as is | PASS | 4/4@0 | 1/6@4 | 1/2@10 (12 IDs) |

Unit checks: all pass.

Receiver transaction IDs, per router: G/ids@start per receiver channel (VC0 | VC1 | VC2); ! = groups
share IDs. Uniform = today's rule; new = assign_transaction_ids. Slots: main's tier pick, dynamic.

| configuration | payload | router | slots | receivers | uniform (today) | new |
| --- | ---: | --- | --- | --- | --- | --- |
| P150 2x4 mesh | 4352 | no deadlock avoidance | tier | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| P150 2x4 mesh | 4352 | no deadlock avoidance | dyn | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| P150 2x4 mesh | 3072 | no deadlock avoidance | tier | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 2x4 mesh | 3072 | no deadlock avoidance | dyn | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 1D line | 4352 | no deadlock avoidance | tier | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 1D line | 4352 | no deadlock avoidance | dyn | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 1D line | 3072 | no deadlock avoidance | tier | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 1D line | 3072 | no deadlock avoidance | dyn | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 1D ring | 4352 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 1D ring | 4352 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 1D ring | 3072 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 1D ring | 3072 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 Y torus (2x4) | 4352 | N/S (wrapped) | tier | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| P150 Y torus (2x4) | 4352 | E/W | tier | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| P150 Y torus (2x4) | 4352 | N/S (wrapped) | dyn | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| P150 Y torus (2x4) | 4352 | E/W | dyn | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| P150 Y torus (2x4) | 3072 | N/S (wrapped) | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 Y torus (2x4) | 3072 | E/W | tier | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 Y torus (2x4) | 3072 | N/S (wrapped) | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 Y torus (2x4) | 3072 | E/W | dyn | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| P150 XY torus (2x4: X ring) | 4352 | deadlock avoidance | tier | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| P150 XY torus (2x4: X ring) | 4352 | deadlock avoidance | dyn | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| P150 XY torus (2x4: X ring) | 3072 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| P150 XY torus (2x4: X ring) | 3072 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 8x4 mesh | 4352 | no deadlock avoidance | tier | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| Galaxy 8x4 mesh | 4352 | no deadlock avoidance | dyn | 16 | 4/16@0 (16 IDs) | 4/4@0 (4 IDs) |
| Galaxy 8x4 mesh | 3072 | no deadlock avoidance | tier | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| Galaxy 8x4 mesh | 3072 | no deadlock avoidance | dyn | 32 | 4/16@0 (16 IDs) | 4/8@0 (8 IDs) |
| Galaxy 1D ring | 4352 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 1D ring | 4352 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 1D ring | 3072 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 1D ring | 3072 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 8x4 XY torus | 4352 | deadlock avoidance | tier | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| Galaxy 8x4 XY torus | 4352 | deadlock avoidance | dyn | 16a | 1/8@0! (8 IDs) | 1/16@0 (16 IDs) |
| Galaxy 8x4 XY torus | 3072 | deadlock avoidance | tier | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| Galaxy 8x4 XY torus | 3072 | deadlock avoidance | dyn | 32a | 1/8@0! (8 IDs) | 1/16@0! (16 IDs) |
| VC1 split (2x 2x2), VC0 first | 4352 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 2x2), VC0 first | 4352 | no deadlock avoidance | dyn | 16 | 4 | 4/4@0 | 1/4@4 (8 IDs) | 4/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 2x2), VC0 first | 3072 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 2x2), VC0 first | 3072 | no deadlock avoidance | dyn | 32 | 6 | 4/4@0! | 1/4@4! (8 IDs) | 4/8@0 | 1/6@8 (14 IDs) |
| VC1 split (2x 2x2), VC1 first | 4352 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 2x2), VC1 first | 4352 | no deadlock avoidance | dyn | 16 | 5 | 4/4@0 | 1/4@4! (8 IDs) | 4/4@0 | 1/5@4 (9 IDs) |
| VC1 split (2x 2x2), VC1 first | 3072 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 2x2), VC1 first | 3072 | no deadlock avoidance | dyn | 32 | 8 | 4/4@0! | 2/4@4 (8 IDs) | 4/8@0 | 2/4@8 (12 IDs) |
| VC1 split (2x 4x4) | 4352 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 4x4) | 4352 | no deadlock avoidance | dyn | 16 | 4 | 4/4@0 | 1/4@4 (8 IDs) | 4/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 4x4) | 3072 | no deadlock avoidance | tier | 8 | 4 | 2/4@0 | 1/4@4 (8 IDs) | 2/4@0 | 1/4@4 (8 IDs) |
| VC1 split (2x 4x4) | 3072 | no deadlock avoidance | dyn | 32 | 6 | 4/4@0! | 1/4@4! (8 IDs) | 4/8@0 | 1/6@8 (14 IDs) |
| VC2 on (P150 2x4, Galaxy 8x4) | 4352 | no deadlock avoidance | tier | 11 | 4 | 2/4@0! | 1/4@4 (8 IDs) | 2/6@0 | 1/4@6 (10 IDs) |
| VC2 on (P150 2x4, Galaxy 8x4) | 4352 | no deadlock avoidance | dyn | 16 | 2 | 4/4@0 | 1/4@4 (8 IDs) | 4/4@0 | 1/2@4 (6 IDs) |
| VC2 on (P150 2x4, Galaxy 8x4) | 3072 | no deadlock avoidance | tier | 11 | 4 | 2/4@0! | 1/4@4 (8 IDs) | 2/6@0 | 1/4@6 (10 IDs) |
| VC2 on (P150 2x4, Galaxy 8x4) | 3072 | no deadlock avoidance | dyn | 32 | 4 | 4/4@0! | 1/4@4 (8 IDs) | 4/8@0 | 1/4@8 (12 IDs) |
| VC1 split + VC2 (three VCs) | 4352 | no deadlock avoidance | tier | 8 | 4 | 4 | 2/4@0 | 1/4@4 | 1/4@8 (12 IDs) | 2/4@0 | 1/4@4 | 1/4@8 (12 IDs) |
| VC1 split + VC2 (three VCs) | 4352 | no deadlock avoidance | dyn | 16 | 4 | 2 | 4/4@0 | 1/4@4 | 1/4@8 (12 IDs) | 4/4@0 | 1/4@4 | 1/2@8 (10 IDs) |
| VC1 split + VC2 (three VCs) | 3072 | no deadlock avoidance | tier | 8 | 4 | 4 | 2/4@0 | 1/4@4 | 1/4@8 (12 IDs) | 2/4@0 | 1/4@4 | 1/4@8 (12 IDs) |
| VC1 split + VC2 (three VCs) | 3072 | no deadlock avoidance | dyn | 32 | 5 | 3 | 4/4@0! | 1/4@4! | 1/4@8 (12 IDs) | 4/8@0 | 1/5@8 | 1/3@13 (16 IDs) |
| Multi-mesh XY torus (VC1, 4x4 meshes) | 4352 | deadlock avoidance | tier | 8a | 4 | 1/8@0 | 1/8@8 (16 IDs) | 1/8@0 | 1/4@8 (12 IDs) |
| Multi-mesh XY torus (VC1, 4x4 meshes) | 4352 | deadlock avoidance | dyn | 16a | 4 | 1/8@0! | 1/8@8 (16 IDs) | 1/8@0! | 1/4@8 (12 IDs) |
| Multi-mesh XY torus (VC1, 4x4 meshes) | 3072 | deadlock avoidance | tier | 8a | 4 | 1/8@0 | 1/8@8 (16 IDs) | 1/8@0 | 1/4@8 (12 IDs) |
| Multi-mesh XY torus (VC1, 4x4 meshes) | 3072 | deadlock avoidance | dyn | 32a | 6 | 1/8@0! | 1/8@8 (16 IDs) | 1/8@0! | 1/6@8 (14 IDs) |
