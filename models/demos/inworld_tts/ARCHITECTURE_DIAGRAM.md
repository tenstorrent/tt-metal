# Inworld TTS - Block-by-Block Architecture Flow

## Full Pipeline Overview

```
                         INFERENCE PIPELINE
 ============================================================================

  [Prompt Audio]              [Text Input]
       |                           |
       v                           v
  +-----------+              +-----------+
  |  ENCODER  |              | Tokenizer |
  | (1x, CPU) |              |  (CPU)    |
  +-----------+              +-----------+
       |                           |
       | speech_ids (VQ codes)     | text_tokens
       v                           v
  +------------------------------------------------+
  |          PROMPT COMPILER (CPU)                  |
  |  "{text}<|speech_start|><|s_N|><|s_N|>..."     |
  +------------------------------------------------+
                      |
                      v
  +------------------------------------------------+
  |          LLaMA 3.1 8B  (TTNN, 2x BH Galaxy)   |
  |  32 layers, GQA, RoPE, SwiGLU, 193K vocab     |
  |  Autoregressive: generates <|s_N|> tokens      |
  |  Until <|speech_end|> or max_tokens=1792       |
  +------------------------------------------------+
                      |
                      | speech token IDs -> integer VQ codes
                      v
  +------------------------------------------------+
  |          DECODER  (TTNN, 1x BH chip)           |
  |  FSQ dequant -> VocosBackbone -> ISTFT         |
  +------------------------------------------------+
                      |
                      v
                 [Audio WAV]
                  16kHz output
```

## Audio Encoder (Detailed)

```
  Audio Waveform [B, 1, 16000] @ 16kHz
  |
  |  =================== ACOUSTIC PATH (CPU) ===================
  |
  v
  +----------------------------------------------------------+
  | ACOUSTIC ENCODER (CodecEnc) - 38.5M params               |
  |                                                          |
  |  Conv1d(1, 48, k=7, pad=3) + WeightNorm                 |
  |  |                                                       |
  |  v                                                       |
  |  EncoderBlock 1: stride=2, 48->96 channels               |
  |  |  3x ResidualUnit:                                     |
  |  |    SnakeBeta(alpha, beta) + AntiAlias(FIR k=12)       |
  |  |    -> WNConv1d(C, C, k=7, pad=3)                     |
  |  |    -> SnakeBeta + AntiAlias                           |
  |  |    -> WNConv1d(C, C, k=1) + residual                 |
  |  |  SnakeBeta + WNConv1d(48, 96, k=4, stride=2)         |
  |  |                                                       |
  |  v                                                       |
  |  EncoderBlock 2: stride=2, 96->192 channels              |
  |  v                                                       |
  |  EncoderBlock 3: stride=4, 192->384 channels             |
  |  v                                                       |
  |  EncoderBlock 4: stride=4, 384->768 channels             |
  |  v                                                       |
  |  EncoderBlock 5: stride=5, 768->1536 channels            |
  |  |                                                       |
  |  v                                                       |
  |  SnakeBeta(1536) + WNConv1d(1536, 1024, k=3, pad=1)     |
  |  |                                                       |
  |  Output: [B, 1024, T]  (T = samples/320 = 50 tok/sec)   |
  +----------------------------------------------------------+
  |
  |  =================== SEMANTIC PATH (TTNN + CPU) ================
  |
  |  Audio Waveform [B, 1, samples]
  |  v
  |  AutoFeatureExtractor (CPU) -> mel filterbank [B, T, 160]
  |  v
  |  +------------------------------------------------------+
  |  | WAV2VEC2-BERT (frozen, ~387M for 16 layers)           |
  |  |  facebook/w2v-bert-2.0, Conformer Encoder             |
  |  |                                                       |
  |  |  Feature Projection (CPU):                            |
  |  |    LayerNorm(160) -> Linear(160, 1024)                |
  |  |    -> [B, T, 1024]                                    |
  |  |                                                       |
  |  |  16x Conformer Layers:                                |
  |  |  +--------------------------------------------------+ |
  |  |  | Conformer Layer (x16)                             | |
  |  |  |                                                   | |
  |  |  | (1) FFN1 -- Macaron half-step:                    | |
  |  |  |   LayerNorm [TTNN L1]                             | |
  |  |  |   Linear(1024,4096) [TTNN L1] -> SiLU [TTNN L1]  | |
  |  |  |   Linear(4096,1024) [TTNN L1]                     | |
  |  |  |   x = h * 0.5 + residual [TTNN L1]               | |
  |  |  |                                                   | |
  |  |  | (2) Self-Attention:                               | |
  |  |  |   LayerNorm [TTNN L1]                             | |
  |  |  |   Q,K,V = 3x Linear(1024,1024) [TTNN L1]         | |
  |  |  |   scores = QK^T/sqrt(64) [host]                   | |
  |  |  |   + relative_key position bias [host]              | |
  |  |  |     distance_embedding [73, 64]                    | |
  |  |  |     clamp distances to [-64, +8]                   | |
  |  |  |   softmax -> @V [host]                             | |
  |  |  |   O = Linear(1024,1024) [TTNN L1]                 | |
  |  |  |   x = h + residual [TTNN L1]                      | |
  |  |  |                                                   | |
  |  |  | (3) Convolution Module:                           | |
  |  |  |   LayerNorm [TTNN L1]                             | |
  |  |  |   PointwiseConv1d(1024,2048,k=1) [TTNN linear]   | |
  |  |  |   GLU [host: split + sigmoid + mul]               | |
  |  |  |   DepthwiseConv1d(1024,k=31,g=1024) [host]       | |
  |  |  |     causal left-pad=30                            | |
  |  |  |   LayerNorm [TTNN L1]                             | |
  |  |  |   SiLU [TTNN L1]                                  | |
  |  |  |   PointwiseConv1d(1024,1024,k=1) [TTNN linear]   | |
  |  |  |   x = h + residual [TTNN L1]                      | |
  |  |  |                                                   | |
  |  |  | (4) FFN2 -- Macaron half-step:                    | |
  |  |  |   (same as FFN1)                                  | |
  |  |  |   x = h * 0.5 + residual [TTNN L1]               | |
  |  |  |                                                   | |
  |  |  | Final LayerNorm [TTNN L1]                         | |
  |  |  +--------------------------------------------------+ |
  |  |                                                       |
  |  |  Output: hidden_states[16] = [B, T, 1024]            |
  |  +------------------------------------------------------+
  |  |
  |  v
  |  +------------------------------------------------------+
  |  | SEMANTIC ENCODER - 12.6M params                       |
  |  |  Conv1d(1024, 1024, k=3, pad=1)                      |
  |  |  -> ReLU + Conv1d + ReLU + Conv1d + residual          |
  |  |  -> Conv1d(1024, 1024, k=3, pad=1)                   |
  |  |  Output: [B, 1024, T]                                 |
  |  +------------------------------------------------------+
  |
  |  =================== FUSION + QUANTIZE =====================
  |
  v                              v
  [Acoustic: B,1024,T]    [Semantic: B,1024,T]
  |                              |
  +----------> CONCAT <----------+
              [B, 2048, T]
                   |
                   v
  +------------------------------------------------------+
  | fc_prior: Linear(2048, 2048)  (TTNN)                 |
  +------------------------------------------------------+
                   |
                   v
  +------------------------------------------------------+
  | FSQ QUANTIZER (CPU)                                   |
  |  project_in: Linear(2048, 8)                          |
  |  -> round to levels [4,4,4,4,4,4,4,4]                |
  |  -> encode to flat index (0..65535)                   |
  |  project_out: Linear(8, 2048)                         |
  +------------------------------------------------------+
                   |
                   v
            VQ codes [B, 1, T]
            (integer indices 0-65535)
```

## LLaMA 8B Backbone (tts_sim target config)

```
  Token IDs [B, seq_len]  (text + speech tokens, vocab=193856)
  |
  v
  +------------------------------------------------------+
  | EMBEDDING: [193856, 4096]                             |
  |  -> [B, seq_len, 4096]                               |
  +------------------------------------------------------+
  |
  |  x32 DECODER LAYERS (pipelined across 2x Galaxy = 64 BH chips)
  |  Each layer = 1x2 BH mesh (TP=2)
  |
  v
  +======================================================+
  | DECODER LAYER (x32)                                   |
  |                                                       |
  |  residual ---+                                        |
  |              |                                        |
  |  RMSNorm ----+---> Attention Block                    |
  |  |                  |                                 |
  |  |    +-------------+-------------+                   |
  |  |    |             |             |                   |
  |  |  Q_proj        K_proj        V_proj                |
  |  |  [4096,4096]   [4096,1024]   [4096,1024]          |
  |  |  (TP: col)     (TP: col)     (TP: col)            |
  |  |    |             |             |                   |
  |  |  RoPE          RoPE           |                    |
  |  |  (llama3)      (llama3)       |                    |
  |  |    |             |             |                   |
  |  |    +-------> GQA SDPA <-------+                    |
  |  |              (causal=True)                         |
  |  |              32 heads, 8 KV heads                  |
  |  |              head_dim=128                          |
  |  |                  |                                 |
  |  |               O_proj [4096,4096] (TP: row)         |
  |  |                  |                                 |
  |  |              AllReduce (TP=2)                       |
  |  |                  |                                 |
  |  +---> ADD <--------+                                 |
  |         |                                             |
  |  residual ---+                                        |
  |              |                                        |
  |  RMSNorm ----+---> MLP Block (SwiGLU)                 |
  |  |                  |                                 |
  |  |    +-------------+-------------+                   |
  |  |    |                           |                   |
  |  |  gate_proj [4096,14336]      up_proj [4096,14336]  |
  |  |  (TP: col, bfp4)            (TP: col, bfp4)       |
  |  |    |                           |                   |
  |  |  SiLU                          |                   |
  |  |    |                           |                   |
  |  |    +---------> MUL <-----------+                   |
  |  |                  |                                 |
  |  |              down_proj [14336,4096] (TP: row)      |
  |  |                  |                                 |
  |  |              AllReduce (TP=2)                       |
  |  |                  |                                 |
  |  +---> ADD <--------+                                 |
  |                                                       |
  +======================================================+
  |
  v
  +------------------------------------------------------+
  | FINAL RMSNorm                                         |
  | LM HEAD: [4096, 193856] (8 shards of 16032)          |
  | -> logits -> sampling (temp=0.8, top_k=50)            |
  +------------------------------------------------------+
  |
  v
  Speech token IDs: <|s_312|> <|s_4091|> <|s_27|> ...
  -> extract integers: [312, 4091, 27, ...]
```

## Audio Decoder (Detailed)

```
  VQ codes [B, T]  (integers 0-65535, T tokens at 50 tok/sec)
  |
  |  =================== FSQ DEQUANTIZE (CPU) ===================
  v
  +------------------------------------------------------+
  | FSQ DEQUANTIZER                                       |
  |  index -> decompose to 8 dims ([4]^8 levels)          |
  |  -> project_out: Linear(8, 2048)                      |
  |  Output: [B, T, 2048]                                 |
  +------------------------------------------------------+
  |
  |  =================== PROJECTION (TTNN) ====================
  v
  +------------------------------------------------------+
  | fc_post_a: Linear(2048, 1024)  [TTNN, L1, full grid]  |
  |  Output: [B, T, 1024]                                 |
  +------------------------------------------------------+
  |
  |  =================== VOCOS BACKBONE (TTNN) =================
  v
  +======================================================+
  | VOCOS BACKBONE - 183.5M params, 12 transformer layers |
  |                                                       |
  |  EMBED: Conv1d(1024, 1024, k=7, pad=3) [TTNN BLOCK_SHARDED] |
  |  |                                                    |
  |  v                                                    |
  |  PRIOR_NET: 2x ResnetBlock                            |
  |  |  Each ResnetBlock:                                 |
  |  |    GroupNorm(32) [host]                             |
  |  |    -> SiLU [TTNN L1]                               |
  |  |    -> Conv1d(1024,1024,k=3,pad=1) [TTNN BLOCK_SHARDED] |
  |  |    -> GroupNorm(32) [host]                          |
  |  |    -> SiLU [TTNN L1]                               |
  |  |    -> Conv1d(1024,1024,k=3,pad=1) [TTNN BLOCK_SHARDED] |
  |  |    -> Residual Add [TTNN L1]                       |
  |  |                                                    |
  |  v                                                    |
  |  12x TRANSFORMER BLOCKS [ALL TTNN L1]                 |
  |  +--------------------------------------------------+ |
  |  | TransformerBlock (bidirectional, NON-CAUSAL)      | |
  |  |                                                   | |
  |  |  residual ---+                                    | |
  |  |              |                                    | |
  |  |  RMSNorm ----+--> Attention [TTNN L1]             | |
  |  |  [TTNN L1]        |                              | |
  |  |                   QKV fused Linear(1024, 3072)    | |
  |  |                   [TTNN L1, full grid]            | |
  |  |                   |                              | |
  |  |                   QKV reshape [host roundtrip]    | |
  |  |                   |                              | |
  |  |                   RoPE (interleaved pairs)        | |
  |  |                   [host, torchtune-compatible]    | |
  |  |                   dim=64, n_positions=16(heads)   | |
  |  |                   |                              | |
  |  |                   SDPA (is_causal=False)          | |
  |  |                   [TTNN L1, 16 heads, hd=64]     | |
  |  |                   |                              | |
  |  |                   Head merge [TTNN permute]       | |
  |  |                   |                              | |
  |  |                   O_proj Linear(1024, 1024)       | |
  |  |                   [TTNN L1, full grid]            | |
  |  |                   |                              | |
  |  |  +---> ADD <------+ [TTNN L1]                    | |
  |  |  |                                               | |
  |  |  residual ---+                                    | |
  |  |              |                                    | |
  |  |  RMSNorm ----+--> MLP (SiLU, NOT SwiGLU)         | |
  |  |  [TTNN L1]        |                              | |
  |  |                   fc1 Linear(1024, 4096)          | |
  |  |                   [TTNN L1, full grid]            | |
  |  |                   |                              | |
  |  |                   SiLU [TTNN L1]                  | |
  |  |                   |                              | |
  |  |                   fc2 Linear(4096, 1024)          | |
  |  |                   [TTNN L1, full grid]            | |
  |  |                   |                              | |
  |  |  +---> ADD <------+ [TTNN L1]                    | |
  |  +--------------------------------------------------+ |
  |  |                                                    |
  |  v                                                    |
  |  POST_NET: 2x ResnetBlock (same as prior_net)         |
  |  |                                                    |
  |  v                                                    |
  |  FINAL LayerNorm [TTNN L1]                            |
  |  Output: [B, T, 1024]                                 |
  +======================================================+
  |
  |  =================== ISTFT HEAD (CPU) =======================
  v
  +------------------------------------------------------+
  | ISTFT HEAD                                            |
  |  Linear(1024, 1282) [CPU, float32]                    |
  |  |                                                    |
  |  Split -> magnitude [641] + phase [641]               |
  |  |                                                    |
  |  magnitude = exp(clamp(mag, max=1e2))                 |
  |  S = mag * (cos(phase) + j*sin(phase))                |
  |  |                                                    |
  |  ISTFT: irfft + window + overlap-add                  |
  |    n_fft=1280, hop_length=320, win_length=1280        |
  |  |                                                    |
  |  Output: [B, 1, num_samples]                          |
  |    num_samples = T * 320 (50 tok/sec * 320 = 16kHz)   |
  +------------------------------------------------------+
  |
  v
  Audio Waveform [B, 1, samples] @ 16kHz
```

## Block Placement Summary

```
  +============================================================+
  |                    ON TENSTORRENT (TTNN)                    |
  +============================================================+
  |                                                             |
  |  DECODER:                                                   |
  |    fc_post_a Linear ............... L1, full core grid       |
  |    Embed Conv1d ................... BLOCK_SHARDED L1         |
  |    ResnetBlock Conv1d ............. BLOCK_SHARDED L1         |
  |    ResnetBlock SiLU ............... L1                       |
  |    ResnetBlock Residual Add ....... L1                       |
  |    TransformerBlock RMSNorm ....... L1                       |
  |    TransformerBlock QKV Linear .... L1, full core grid       |
  |    TransformerBlock SDPA .......... L1                       |
  |    TransformerBlock O Linear ...... L1, full core grid       |
  |    TransformerBlock MLP fc1 ....... L1, full core grid       |
  |    TransformerBlock MLP SiLU ...... L1                       |
  |    TransformerBlock MLP fc2 ....... L1, full core grid       |
  |    TransformerBlock Residual Add .. L1                       |
  |    Final LayerNorm ................ L1                       |
  |                                                             |
  |  ENCODER:                                                   |
  |    Wav2Vec2-BERT FFN1/FFN2 Linear . L1, full core grid       |
  |    Wav2Vec2-BERT FFN SiLU ......... L1                       |
  |    Wav2Vec2-BERT Q/K/V/O Linear ... L1, full core grid       |
  |    Wav2Vec2-BERT LayerNorm (x5) ... L1                       |
  |    Wav2Vec2-BERT PointwiseConv .... L1 (as ttnn.linear)      |
  |    Wav2Vec2-BERT Macaron x0.5 ..... L1                       |
  |    Wav2Vec2-BERT Residual Add ..... L1                       |
  |    fc_prior Linear ................ L1, full core grid       |
  |                                                             |
  |  LLM (target: 2x Galaxy, 64 BH chips):                     |
  |    32 decoder layers .............. SRAM, TP=2 per layer     |
  |    Embedding ...................... DRAM                     |
  |    LM Head ....................... DRAM-sharded              |
  |                                                             |
  +============================================================+
  |                    ON HOST CPU                               |
  +============================================================+
  |                                                             |
  |  DECODER:                                                   |
  |    FSQ dequantize ................. codebook lookup          |
  |    ResnetBlock GroupNorm .......... host roundtrip           |
  |    Attention RoPE ................. host roundtrip (5D)      |
  |    Attention QKV reshape .......... host roundtrip           |
  |    ISTFTHead Linear ............... float32 for FFT          |
  |    ISTFTHead ISTFT (FFT) ......... torch.fft                |
  |                                                             |
  |  ENCODER:                                                   |
  |    AcousticEncoder ................ Conv1d + SnakeBeta       |
  |    Wav2Vec2-BERT FeatureProj ...... 160->1024, not tile-aligned |
  |    Wav2Vec2-BERT Attn + PosBias ... SDPA no additive bias support |
  |    Wav2Vec2-BERT DepthwiseConv .... groups=1024, k=31       |
  |    Wav2Vec2-BERT GLU .............. split + sigmoid + mul   |
  |    SemanticEncoder ................ Conv1d residual          |
  |    FSQ quantize ................... codebook + rounding      |
  |                                                             |
  +============================================================+
```

## Performance Summary

```
  +============================================================+
  |  MEASURED PERFORMANCE (Blackhole P150, single chip)         |
  +============================================================+
  |                                                             |
  |  DECODER (100 tokens = 2s audio):                           |
  |    VocosBackbone .......... 19.8ms (ref: 58.5ms) = 3.0x     |
  |    Full Decoder ........... 21.9ms (ref: 60.3ms) = 2.8x     |
  |    Backbone PCC ........... 0.9704                          |
  |    Device time (tracy) .... 9.2ms (92% L1, 8% DRAM)        |
  |    Host overhead .......... ~11ms (GroupNorm + RoPE)        |
  |                                                             |
  |  ENCODER (1s audio = 50 tokens):                            |
  |    Wav2Vec2-BERT (16 layers) . ~152ms ref (387M params)     |
  |    AcousticEncoder ........... ~125ms ref (38.5M params)    |
  |    SemanticEncoder ........... ~3ms ref (12.6M params)      |
  |    Fusion + FSQ .............. ~2ms ref                     |
  |    Full Encoder (excl w2v) ... 101.6ms (ref: 125.7ms)=1.2x |
  |    (Wav2Vec2-BERT: FFN+LN on TTNN, attn+conv on host)      |
  |                                                             |
  |  LLM (1 layer, batch=1, from pytest):                       |
  |    Per-layer decode ....... 1.175ms (TP=1, DRAM weights)    |
  |    Total 1-layer .......... 36.56ms (incl LM head)          |
  |    tts_sim target ......... 14.9us/layer (TP=2, SRAM)       |
  |                                                             |
  +============================================================+
  |  tts_sim PIPELINE TARGETS (2x Galaxy, TP=2, SRAM, FP8):    |
  +============================================================+
  |    Per-layer decode ....... 14.9us (Impl, 50% SoL)          |
  |    32-layer TPOT .......... 477us                            |
  |    First chunk (30 tok) ... 150ms P50 target                 |
  |    Subsequent chunk ....... <270ms deadline                  |
  |    Throughput ............. 66,900 agg tok/s (SoL)           |
  +============================================================+
```

## Audio Constants

```
  Sample rate:        16,000 Hz
  Token rate:         50 tokens/sec
  Hop length:         320 samples (16000 / 50)
  FSQ levels:         [4,4,4,4,4,4,4,4]
  FSQ codebook size:  65,536 (4^8)
  FSQ num_quantizers: 1
  ISTFT n_fft:        1,280
  ISTFT hop_length:   320
  ISTFT win_length:   1,280
  Audio chunk:        30 tokens = 0.6s audio
  Max output tokens:  1,792
```
