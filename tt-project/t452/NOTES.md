
## t452 run 1710 (2026-10-12 11:45 PDT) — isolation: our SLA attention mode is the cause
- Chain 1709 on blx01 (900 MHz ceiling, relative only): job 023 `slalora` PASS, job 024 `v10sla0.85` PASS. No drops.
- slalora (SLA LoRA v0.1, dense attention): 3105 ms/step (same as dense), audio peak 0.041 rms 0.004 0% clipped, sane fox video.
  vs v1.0 dense: Y PSNR 18.18 dB, latent PCC 0.894 (different LoRA, so drift, not error).
- v10sla0.85 (v1.0 LoRA + MINIMAX_H3_VSA_MODE=sla, sparsity 0.85): 2451 ms/step, Y PSNR 12.00 dB, latent PCC 0.295,
  audio peak 1.0 rms 0.43 2.6% clipped: the same streak noise as SLA LoRA + sla.
- Still: stills/iso-s0.jpg (left dense v1.0, middle SLA LoRA dense attn, right v1.0 + sla mode).
- Conclusion: the noise comes from our SLA selection mode (sla.patch: vsa_stages_minimax_h3.py, vsa_geometry.py, pipeline),
  not from the LoRA. Either the block selection/mask is wrong (e.g. layout, token order or audio tokens included in the
  sparse path), or SLA cannot work without its linear-attention branch for the dropped blocks. The 20% speed number stands
  only as an upper bound until the mode produces sane output.
- Recommendation: do not ship SLA or VSA on Turbo now. Next: a host CPU test comparing our sla-mode attention output on one
  layer's real Q/K/V against a torch SLA reference (lightx2v's SLA: top-k block sparse + linear attention on the rest);
  fix, then rerun s0 sla0.85 vs dense. VBench subset skipped: noise output makes it meaningless.
