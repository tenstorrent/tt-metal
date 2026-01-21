# BGE-Large-EN-v1.5 PCC Tests - Complete Suite

## ✅ All PCC Tests Created

Complete set of PCC tests to debug where lower PCC values are coming from. Each test validates a specific component of the BGE model.

### 📁 Test Files (10 total)

1. **`test_ttnn_bge_embeddings.py`** - Tests embeddings layer
   - Input: `[8, 384]` (batch, seq_len)
   - Output: `[8, 384, 1024]` (batch, seq_len, hidden_size)
   - Grid: (8, 8)
   - PCC Target: 0.99

2. **`test_ttnn_bge_self_attention.py`** - Tests self-attention mechanism
   - Input: `[8, 384, 1024]` (batch, seq_len, hidden_size)
   - Attention mask: `[8, 1, 1, 384]`
   - Grid: (8, 8)
   - PCC Target: 0.99

3. **`test_ttnn_bge_self_output.py`** - Tests attention output projection
   - Input: `[8, 384, 1024]` (hidden_states, input_tensor)
   - Grid: (8, 8)
   - PCC Target: 0.99

4. **`test_ttnn_bge_attention.py`** - Tests full attention module (self + output)
   - Input: `[8, 384, 1024]`
   - Grid: (8, 8)
   - PCC Target: 0.99

5. **`test_ttnn_bge_intermediate.py`** - Tests FFN intermediate (1024 → 4096)
   - Input: `[8, 384, 1024]`
   - Output: `[8, 384, 4096]`
   - Grid: (8, 8)
   - PCC Target: 0.99

6. **`test_ttnn_bge_output.py`** - Tests FFN output (4096 → 1024)
   - Input: `[8, 384, 4096]` (hidden_states), `[8, 384, 1024]` (input_tensor)
   - Output: `[8, 384, 1024]`
   - Grid: (8, 8)
   - PCC Target: 0.99

7. **`test_ttnn_bge_layer.py`** - Tests complete transformer layer
   - Input: `[8, 384, 1024]`
   - Grid: (8, 8)
   - PCC Target: 0.99

8. **`test_ttnn_bge_encoder.py`** - Tests all 24 encoder layers
   - Input: `[8, 384, 1024]`
   - Grid: (8, 8)
   - PCC Target: 0.98 (lower due to 24 layers)

9. **`test_ttnn_bge_pooler.py`** - Tests pooler layer
   - Input: `[8, 384, 1024]`
   - Grid: N/A (uses interleaved memory)
   - PCC Target: 0.99

10. **`test_ttnn_bge_model.py`** - Tests full model end-to-end
    - Input: `[8, 384]` (input_ids)
    - Output: Mean pooled embeddings `[8, 1024]`
    - PCC Target: 0.95 (as set in test)

## 🚀 Running Tests

### Run Individual Tests
```bash
# Test embeddings
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_embeddings.py::test_ttnn_bge_embeddings

# Test self-attention
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_self_attention.py::test_ttnn_bge_self_attention

# Test self-output
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_self_output.py::test_ttnn_bge_self_output

# Test attention module
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_attention.py::test_ttnn_bge_attention

# Test intermediate
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_intermediate.py::test_ttnn_bge_intermediate

# Test output
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_output.py::test_ttnn_bge_output

# Test layer
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_layer.py::test_ttnn_bge_layer

# Test encoder
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_encoder.py::test_ttnn_bge_encoder

# Test pooler
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_pooler.py::test_ttnn_bge_pooler

# Test full model
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_model.py::test_ttnn_bge_model
```

### Run All PCC Tests
```bash
pytest --disable-warnings models/demos/wormhole/bge_large_en/tests/pcc/
```

## 🔍 Debugging Strategy

Run tests in order to identify where PCC drops:

1. **Start with embeddings** - Should be 0.99+
   - If low: Check embedding weights loading

2. **Test self-attention** - Should be 0.99+
   - If low: Check QKV fusion, attention computation

3. **Test self-output** - Should be 0.99+
   - If low: Check dense layer, layer norm

4. **Test attention module** - Should be 0.99+
   - If low: Check integration of self-attention + output

5. **Test intermediate** - Should be 0.99+
   - If low: Check FFN first part (1024 → 4096), GELU activation

6. **Test output** - Should be 0.99+
   - If low: Check FFN second part (4096 → 1024), layer norm

7. **Test layer** - Should be 0.99+
   - If low: Check integration of attention + FFN

8. **Test encoder** - Should be 0.98+
   - If low: Check error accumulation across 24 layers

9. **Test pooler** - Should be 0.99+
   - If low: Check pooler layer (usually fine, uses interleaved memory)

10. **Test full model** - Should be 0.95+
    - If low: Check mean pooling, end-to-end integration

## 📊 Expected PCC Values

| Component | Expected PCC | Notes |
|-----------|--------------|-------|
| Embeddings | 0.99+ | First layer, should be high |
| Self-Attention | 0.99+ | Core attention mechanism |
| Self-Output | 0.99+ | Attention projection |
| Attention Module | 0.99+ | Combined attention |
| Intermediate | 0.99+ | FFN first part |
| Output | 0.99+ | FFN second part |
| Layer | 0.99+ | Single transformer layer |
| Encoder | 0.98+ | 24 layers, may accumulate error |
| Pooler | 0.99+ | Usually fine |
| Full Model | 0.95+ | End-to-end with mean pooling |

## 🔧 Key Differences from SentenceBERT

| Aspect | SentenceBERT | BGE-Large |
|--------|--------------|-----------|
| **Hidden Size** | 768 | 1024 |
| **Grid** | (6, 8) | (8, 8) |
| **Layers** | 12 | 24 |
| **Intermediate** | 3072 | 4096 |
| **Input Shapes** | [8, 384, 768] | [8, 384, 1024] |

## 🎯 Troubleshooting

If PCC is lower than expected:

1. **Check grid sizes** - All should use (8, 8) for BGE
2. **Check config imports** - All should import from `bge_large_en` common
3. **Check model loading** - Should load `BAAI/bge-large-en-v1.5`
4. **Check program configs** - May need tuning for larger dimensions
5. **Check numerical precision** - bfloat8_b may cause precision loss

Run tests sequentially to pinpoint where PCC drops! 🔍
