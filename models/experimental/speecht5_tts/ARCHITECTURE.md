# SpeechT5 TTS Architecture Documentation

## Model Overview

SpeechT5 is an encoder-decoder transformer model for text-to-speech (TTS) that converts text inputs into mel-spectrograms, which can then be converted to audio using a vocoder.

## Model Configuration

- **Hidden size**: 768
- **Vocab size**: 81
- **Encoder layers**: 12
- **Decoder layers**: 6
- **Encoder attention heads**: 12
- **Decoder attention heads**: 12
- **Encoder FFN dim**: 3072
- **Decoder FFN dim**: 3072
- **Num mel bins**: 80
- **Reduction factor**: 2 (outputs 2 mel frames per decoder step)

## Architecture Components

### 1. Text Encoder Pre-Net

**Purpose**: Converts input text tokens to embeddings

**Components**:
- `embed_tokens`: Embedding layer (vocab_size=81, hidden_size=768)
- `encode_positions`: Scaled positional encoding with dropout

**Input**: Text token IDs [batch, seq_len]
**Output**: Embedded text [batch, seq_len, 768]

### 2. Encoder (12 layers)

**Purpose**: Processes text embeddings into contextualized representations

**Components per layer**:
- **Self-Attention Block**:
  - Q, K, V projections: Linear(768, 768) each with bias
  - Output projection: Linear(768, 768) with bias
  - 12 attention heads (head_dim = 64)
  - Layer norm before attention
  - Dropout (p=0.1)

- **Feed-Forward Block**:
  - Intermediate dense: Linear(768, 3072) with bias
  - Activation: GELU
  - Output dense: Linear(3072, 768) with bias
  - Dropout (p=0.1)
  - Layer norm before FFN

- **Relative Positional Encoding**:
  - `embed_positions.pe_k`: Embedding(320, 64)
  - Applied in self-attention

- **Final Layer Norm**: LayerNorm(768) after all layers

**Input**: Embedded text [batch, seq_len, 768]
**Output**: Encoded text [batch, seq_len, 768]

### 3. Speech Decoder Pre-Net

**Purpose**: Processes previous mel-spectrograms and speaker embeddings

**Components**:
- `layers[0]`: Linear(80, 256) - first prenet layer
- `layers[1]`: Linear(256, 256) - second prenet layer
- `final_layer`: Linear(256, 768) - project to hidden dim
- `encode_positions`: Scaled positional encoding with dropout
- `speaker_embeds_layer`: Linear(1280, 768) - speaker embedding projection

**Input**:
- Mel-spectrogram frames [batch, mel_seq_len, 80]
- Speaker embeddings [batch, 1280]

**Output**: Preprocessed decoder input [batch, mel_seq_len, 768]

### 4. Decoder (6 layers)

**Purpose**: Generates mel-spectrogram features autoregressively

**Components per layer**:
- **Self-Attention Block**:
  - Q, K, V projections: Linear(768, 768) each with bias
  - Output projection: Linear(768, 768) with bias
  - 12 attention heads (head_dim = 64)
  - Causal masking for autoregressive generation
  - Layer norm before attention

- **Cross-Attention Block** (encoder-decoder attention):
  - Q projection (from decoder): Linear(768, 768) with bias
  - K, V projections (from encoder): Linear(768, 768) each with bias
  - Output projection: Linear(768, 768) with bias
  - 12 attention heads (head_dim = 64)
  - Layer norm before cross-attention

- **Feed-Forward Block**:
  - Intermediate dense: Linear(768, 3072) with bias
  - Activation: GELU
  - Output dense: Linear(3072, 768) with bias
  - Dropout (p=0.1)
  - Layer norm before FFN

**Input**:
- Preprocessed mel features [batch, mel_seq_len, 768]
- Encoder output [batch, text_seq_len, 768]

**Output**: Decoded features [batch, mel_seq_len, 768]

### 5. Speech Decoder Post-Net

**Purpose**: Converts decoded features to mel-spectrograms and stop predictions

**Components**:
- **Feature Output**:
  - `feat_out`: Linear(768, 160) - outputs 2 mel frames (80 * 2 = 160)
  - Handles reduction factor of 2

- **Stop Prediction**:
  - `prob_out`: Linear(768, 2) - binary stop prediction

- **Post-Processing Layers** (5 Conv1D layers):
  - Layer 0: Conv1d(80, 256, kernel=5, padding=2) + BatchNorm + Tanh + Dropout(0.5)
  - Layers 1-3: Conv1d(256, 256, kernel=5, padding=2) + BatchNorm + Tanh + Dropout(0.5)
  - Layer 4: Conv1d(256, 80, kernel=5, padding=2) + BatchNorm + Dropout(0.5)

**Input**: Decoded features [batch, mel_seq_len, 768]
**Output**:
- Mel-spectrogram [batch, mel_seq_len * reduction_factor, 80]
- Stop predictions [batch, mel_seq_len, 2]

## Data Flow

```
Text Input [batch, text_len]
    ↓
Text Encoder Pre-Net
    ↓
Encoder (12 layers) with Self-Attention
    ↓
Encoded Text [batch, text_len, 768]
    ↓ (cross-attention input)
    |
    ├─→ Decoder Cross-Attention (KV)
    |
Previous Mel + Speaker → Speech Decoder Pre-Net
                              ↓
                    Decoder Self-Attention
                              ↓
                    Decoder Cross-Attention (Q from here, KV from encoder)
                              ↓
                    Decoder Feed-Forward (6 layers)
                              ↓
                    Speech Decoder Post-Net
                              ↓
        Mel-Spectrogram [batch, mel_len, 80] + Stop Predictions
```

## Key Differences from T5

While SpeechT5 shares similarities with T5 encoder architecture, key differences include:

1. **Decoder Input**: Uses mel-spectrogram features instead of text embeddings
2. **Cross-Attention**: Decoder has cross-attention layers to attend to encoder output
3. **Pre-Net**: Speech decoder has a specialized pre-net for processing mel features
4. **Post-Net**: Has a convolutional post-net for refining mel-spectrograms
5. **Speaker Embeddings**: Supports speaker conditioning through speaker embeddings
6. **Reduction Factor**: Outputs multiple mel frames per decoder step
7. **Positional Encoding**: Uses scaled positional encoding instead of relative position bias

## Parameter Count

- **Encoder**: ~257 parameters (embedding + 12 layers)
- **Decoder**: ~139 parameters (prenet + 6 layers with self-attn)
- **Cross-Attention**: ~60 parameters (6 layers)
- **Post-Net**: ~34 parameters (conv layers + batch norms)
- **Total**: ~490 parameter tensors

## Reusable Components from Existing T5 Implementation

From `models/experimental/stable_diffusion_35_large/`:

1. **Encoder Attention**: Similar self-attention mechanism can be adapted
2. **Feed-Forward**: Same GELU-based FFN structure
3. **Layer Normalization**: Can reuse LayerNorm implementation
4. **Linear Projections**: Same linear layer patterns
5. **Parameter Loading**: Can adapt `from_torch()` pattern

## Components Requiring New Implementation

1. **Speech Decoder Pre-Net**: Mel-spectrogram preprocessing
2. **Cross-Attention**: Encoder-decoder attention mechanism
3. **Convolutional Post-Net**: Conv1D + BatchNorm layers
4. **Autoregressive Decoder**: Causal masking for sequential generation
5. **Speaker Embedding Integration**: Speaker conditioning
6. **Stop Prediction**: Binary classification for sequence end
7. **Reduction Factor Handling**: Multiple frame generation per step
