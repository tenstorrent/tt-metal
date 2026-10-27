# Round 2 proposals (ranked)

| id | candidate | expected | risk | measure |
|---|---|---|---|---|
| R2-1 | Fused sLSTM step kernel (ops/slstm_step, kernel session) once KERNEL.md says DONE. Its v2 at 60.7 us/step x 552 = ~34 ms vs ~294 now | -250..-270 ms (-25 %) | medium (jobs 728/795 hung; chip wedges) | b2b paper/acc + guards, full suite, check_gates both precisions |
| R2-2 | Op-level step restructure: gate-major WxT (drop 4 gate slices), per-head batched recurrent matmul [4,B,128]x[4,128,512], fused activations in binary_ng, `pre` add folded into matmul, step temporaries in L1. Fallback / independent of R2-1 | -55..-100 ms | low-medium (fp32 exactness of fused activations) | step micro-bench, then b2b + check_gates |
| R2-3 | slstm_post per-head norm via ttnn.layer_norm on a head-split view (as mLSTM, layers.py:380) instead of the two 512x512 averaging matmuls + 4 binaries; drop the zero-tail concat (R1-5) | -60..-80 ms (-7 %) | low | b2b + pcc_spread / row floors, both precisions |
| R2-4 | sLSTM pre/post permutes: fold transpose into w_in output ordering / untransposed concat | -30..-57 ms paper, up to -330 ms multivariate | medium | transpose micro-bench, b2b all shapes |
| R2-5 | R1-2d on multivariate: fold W_in bias only on the pad-skip path, separate add on full/variate path | 0..-170 ms multivariate | low | multivariate b2b x2 both precisions |
| R2-6 | mLSTM chunkwise bundle: rest of R1-4, one [512,128] gate linear instead of 4 + concat, trim tilize/untilize/slice | -20..-40 ms | low (bundle to reach 2 %) | b2b + suite |
| R2-7 | Host R1-7: vectorized prepare / postprocess; first a wall-vs-device bench at HEAD | wall -0.3..-0.5 s | low | bench wall |

Not proposed: R1-3 read-out cut (-1.2 % paper, 38 ms multivariate), R1-2e (rejected), R1-4a alone (-0.6 %).
Order: R2-3, R2-2 (while the kernel finishes), R2-5, R2-4, R2-6, R2-7; R2-1 as soon as KERNEL.md is DONE.
Note for the orchestrator: the kernel's 40 us/step gate predates the pad cut (552 steps, not 1710); at 60.7 us it is
already ~-260 ms in model. Re-basing that gate is the kernel session's / orchestrator's call, not taken here.
Profile at HEAD is blocked by the disk watermark; rounds 2 measurements use b2b benches (bench kind is allowed).
