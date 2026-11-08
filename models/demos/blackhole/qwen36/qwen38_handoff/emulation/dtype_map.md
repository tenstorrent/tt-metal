# TT weight dtype map (Qwen3.6/3.8-27B, P150x4 TP=4), emulated by emul_run.py

Quantizer: tt_metal/impl/data_format/blockfloat_common.cpp (pack_as_bfp_tiles / convert_u32_to_bfp<..,truncate=false>),
validated bit-exact vs ttnn.from_tensor(...bfloat8_b/bfloat4_b) -> to_torch (see val.log).
- Block = 16 elements = one FACE ROW (row of a 16x16 face), i.e. 16 contiguous elements along the LAST dim.
  For a TT weight [K=in, N=out] the block runs along N (output dim). get_max_exp: blockfloat_common.cpp:20-43 (exp = fp32 exponent field, max over the 16; is_exp_a=false for *_b formats => no rebias).
  Tile iteration (faces tr,tc, rows i, 16 cols j) at blockfloat_common.cpp:~362-390.
- Mantissa: bfp8 = sign + 7 bits (hidden 1 incl.), bfp4 = sign + 3 bits (MANTISSA_BFP_WIDTH, blockfloat_common.cpp:~253-262).
- Per element: fp32 24-bit mantissa with hidden 1 is first shifted RIGHT by (shared_exp - exp) with TRUNCATION (:~296-306),
  then rounded to nearest-even on the remaining bits (round_value > tie or tie&odd => +1; :~320-331), saturating at 2^w-1 (no overflow carry to the exponent).
  Sign cleared if mantissa==0. Zeros and fp32 subnormals (exp field 0) -> 0 and ignored for max exp (:~280). No exponent rebias for _b.
  Value = m * 2^(shared_exp-127) * 2^-(w-1). (Elements much smaller than block max flush to 0.)
- Sharding (TP=4 column/row-parallel) never splits a 16-block (all shard widths multiples of 16 along N), so emulating on the full [K,N] is exact,
  EXCEPT the fused GDN [qkv|z|a|b] weight: per-device a_d(12)+b_d(12) columns sit at offset 4096 in the device block, so one block holds a[0:12]+b[0:4] and the next b[4:12]+pad.
  Emulated exactly: per device d, quantize cat([a_d,b_d,zeros(8)], dim=N) as [K,32].
- Orientation: HF [out,in] -> quantize w.T as [K=in,N=out] -> transpose back. Stored as bf16 (exact for bfp8/bfp4 values).
- All tensors are cast to bf16 before conversion in TT (tp_common.shard_w); HF weights already bf16, so no-op.
- q/k/v/gate/up/qkv column fusions (prepare_attn_qkv*, prepare_gdn_qkv, swiglu interleave in tile-wide (32) units) only permute/concat whole >=32-wide column groups => block membership unchanged.
  (HF q_proj has [q_h|gate_h] rows per head, 256 each; blocks stay within them.)

| HF tensor | TT storage | Q8 (Tier-2 off) | Q4 (Tier-2 defaults) | Q4all |
|---|---|---|---|---|
| mlp.gate_proj, mlp.up_proj (tt/mlp.py:41,93,101,122,130,156,158) | bfp4_b | BFP4 | BFP4 | BFP4 |
| mlp.down_proj (mlp.py:109,138,157; QWEN36_BFP4_MLP_DOWN, wt-dshard tt/mlp.py:185) | bfp8_b | BFP8 | BFP4 | BFP4 |
| linear_attn.in_proj_qkv, in_proj_z (gdn/tp.py:~159-176) | bfp8_b (BFP4 if QWEN36_BFP4_GDN_IN=1, default off) | BFP8 | BFP8 | BFP4 |
| linear_attn.in_proj_a, in_proj_b (fused into qkvz, gdn/tp.py:~140-160) | same as in_proj_qkv | BFP8 | BFP8 | BFP4 |
| linear_attn.out_proj (gdn/tp.py:~197,208) | bfp8_b | BFP8 | BFP8 | BFP8 |
| self_attn.q_proj (incl. output gate), k_proj, v_proj (attention/tp.py:71,85,95; QWEN36_BFP4_ATTN default 1 in wt-dshard) | bfp8_b | BFP8 | BFP4 | BFP4 |
| self_attn.o_proj (attention/tp.py:103; same flag) | bfp8_b | BFP8 | BFP4 | BFP4 |
| lm_head.weight (model.py:142, vocab-sharded) | bfp8_b | BFP8 | BFP8 | BFP8 |
| embed_tokens (Embedding dtype=bfloat16, model.py:66-72) | bf16 | none | none | none |
| *norm.weight, q_norm/k_norm (stored as 1+w in fp32 then bf16), A_log, dt_bias, conv1d (shard_small/replicate bf16) | bf16 | none (q/k_norm 1+w bf16 rounding not emulated) | | |
Vision tower / MTP: not used.
