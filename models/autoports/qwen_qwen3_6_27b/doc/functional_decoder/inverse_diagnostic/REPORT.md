# Real-input chunk inverse discriminator

The power-product inverse is **not supported as the cause of the length-511 contribution failure by this CPU discriminator**. It is less accurate than HF row substitution, but its errors on this input are small: replacing it changes the complete decoder contribution with minimum per-row PCC **0.9999974874**, comfortably above 0.995. This does not simulate Blackhole operand intake precision or use captured TT intermediates, so it does not exclude a device-specific amplification.

Command (CPU only, exit 0):

```bash
timeout 600 python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/inverse_diagnostic/discriminate.py > models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/inverse_diagnostic/run.log 2>&1
```

The script imports neither TTNN nor autoport runtime modules. It loads pinned revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`, sets eight CPU threads and seed 183, constructs the real BF16 HF layer then rotary embedding, consumes the initial `randperm(64)`, and generates lengths 1, 31, 32, 33, 127, 128, 129, 511 in that order. Setup-only zeros/ones/eye in the TT loader do not consume RNG. HF eval forwards contain no stochastic operation and therefore need not be executed to reconstruct the later inputs. The final input is saved in `input_length511.pt`; its raw BF16 byte SHA256 is `50a69ebec5278c4e507cc340f4b5e50559d8946fca5d7546b3295715465bdc55`.

HF input normalization, projection, causal convolution, SiLU, beta and decay supply identical upstream tensors to both variants. Q/K use the current TT implementation's FP32 L2 formula. Both variants use identical 32-token chunks, masks, recurrence/state equations, and zero padding. Only inverse construction differs. Each inverse is independently compared to FP64 triangular solve. Complete decoder outputs use identical HF gated norm, output projection, residual, post-attention norm and MLP after the alternate recurrent cores.

| Measurement | Power product | HF row substitution |
|---|---:|---:|
| Maximum FP64-evaluated inverse residual | 2.91794578e-6 | 7.34446168e-8 |
| Maximum inverse error versus FP64 solve | 3.22076645e-6 | See per-chunk results |
| Largest matrix power magnitude | 16.87513351 | N/A |
| Largest intermediate product magnitude | 6.64951706 | N/A |

The worst power-product inverse occurs in chunk starting at token 192, expanded value head 10, local row 31, column 3 (absolute query row 223). Next-largest residuals are in chunks 288, 480, 416 and 256. The worst inverse is therefore neither the final tail nor a row immediately adjacent to it. All 16 chunk records include decay/beta bounds and inverse/output/state errors in `results.json`.

Across the full length-511 recurrence, power versus row core-output relative L2 is `1.79334833e-7`, maximum absolute core error is `1.04308128e-7`, and final-state maximum error is `7.15255737e-7`. After BF16 conversion and identical downstream decoder processing, minimum contribution PCC is `0.9999974874` at token 394. That is the worst row of this controlled inverse-only comparison, **not** a recovered location of the original TT failure. The original saved contract result contains no bad-row index or actual TT intermediate tensor, so no defensible link to its failed row can be made.

Artifacts: `discriminate.py` is the reproducible analysis; `run.log` contains command output; `results.json` contains all per-chunk metrics and all 511 contribution PCCs; `input_length511.pt` contains the reconstructed input for an on-device discriminator. CPU arithmetic remains an explicit limit: upstream TT norm/projection/convolution and TT FP32 matrix-multiply intake may differ from these HF/CPU tensors.
