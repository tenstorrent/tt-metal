# GPT-OSS: Mixture of Experts Language Models

**High-performance inference for GPT-OSS models on Tenstorrent Wormhole devices**

---

## 🚀 Quick Start

### What You Need
- **Hardware**: Tenstorrent Wormhole devices (minimum 8 devices recommended)
- **Models**: GPT-OSS-20B or GPT-OSS-120B
- **Environment**: Python with TT-NN framework

### Running Your First Inference

```bash
# Navigate to the demo directory
cd tt-metal/models/demos/gpt_oss/demo

# Run the interactive text generation demo
pytest simple_text_demo.py::test_gpt_oss_demo -v -s
```

**That's it!** The demo will automatically:
- Download and setup the model (if not already available)
- Configure your devices for optimal performance
- Generate text responses to your prompts

---

## 📝 Example Usage

### Basic Text Generation
The demo responds to questions and prompts. Try asking:

- **"How many r's in the word 'strawberry'?"** - Tests reasoning
- **"Write a short story about space exploration"** - Creative writing
- **"Explain quantum computing in simple terms"** - Technical explanations
- **"What are the benefits of renewable energy?"** - Informational responses

### Sample Output
```
Input: "How many r's in the word 'strawberry'?"

Output: "To count the r's in 'strawberry', I'll go through each letter:
s-t-r-a-w-b-e-r-r-y

The letter 'r' appears 3 times in the word 'strawberry':
- Position 3: r
- Position 8: r
- Position 9: r

So there are 3 r's in 'strawberry'."
```

---

## ⚙️ Configuration Options

### Model Selection
Set which model to use via environment variable:
```bash
# For GPT-OSS-20B (faster, good quality)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"

# For GPT-OSS-120B (slower, higher quality)
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-120B"
```

### Device Configuration
The system automatically detects your hardware and configures:
- **Tensor Parallel**: Splits model across devices for speed
- **Expert Parallel**: Distributes MoE experts efficiently
- **Memory Optimization**: Minimizes memory usage per device

### Performance Tuning
Common configurations for different setups:

| **Setup** | **Devices** | **Batch Size** | **Sequence Length** | **Best For** |
|-----------|-------------|----------------|---------------------|--------------|
| **Development** | 8 devices | 1 | 512 tokens | Testing, debugging |
| **Demo** | 8 devices | 1 | 1024 tokens | Interactive use |
| **Production** | 32 devices | 4-8 | 2048 tokens | High throughput |

---

## 🎯 Model Capabilities

### GPT-OSS-20B
- **Parameters**: 20 billion parameters
- **Experts**: 64 MoE experts
- **Speed**: ~25-30 tokens/second
- **Memory**: ~8GB per device
- **Best for**: Fast responses, development, testing

### GPT-OSS-120B
- **Parameters**: 120 billion parameters
- **Experts**: 128 MoE experts
- **Speed**: ~15-20 tokens/second
- **Memory**: ~12GB per device
- **Best for**: Highest quality, complex reasoning

### Key Features
- ✅ **Mixture of Experts**: Only activates relevant experts per token
- ✅ **Multi-head Attention**: Advanced attention mechanisms
- ✅ **Rotary Embeddings**: State-of-the-art position encoding
- ✅ **Instruction Following**: Optimized for chat and Q&A
- ✅ **Long Context**: Supports up to 4K+ token sequences

---

## 📊 Performance Expectations

### Typical Performance (8 Wormhole devices)

| **Model** | **Time to First Token** | **Generation Speed** | **Memory Usage** |
|-----------|-------------------------|---------------------|------------------|
| **GPT-OSS-20B** | 150-200ms | 25-30 tok/s | ~64GB total |
| **GPT-OSS-120B** | 200-300ms | 15-20 tok/s | ~96GB total |

### Scaling Performance
- **More devices** = Faster inference (up to 32+ devices supported)
- **Larger batch sizes** = Higher throughput
- **Longer sequences** = Better context understanding

---

## 🔧 Advanced Usage

### Custom Prompts
Edit the demo to use your own prompts:
```python
input_prompts = ["Your custom prompt here"]
```

### Generation Parameters
Adjust text generation behavior:
```python
sampling_params = {
    "temperature": 0.7,    # Higher = more creative (0.0 = deterministic)
    "top_p": 0.9,          # Nucleus sampling threshold
}
max_generated_tokens = 500  # Maximum response length
```

### Multi-User Batching
Process multiple prompts simultaneously:
```python
batch_size = 4  # Process 4 prompts at once
input_prompts = [
    "Question 1...",
    "Question 2...",
    "Question 3...",
    "Question 4..."
]
```

---

## 🛠️ Troubleshooting

### Common Issues

**❌ "Model not found" error**
```bash
# Solution: Set the correct model path
export GPT_DIR="/mnt/MLPerf/tt_dnn-models/tt/GPT-OSS-20B"
```

**❌ "Out of memory" error**
- Try reducing `max_seq_len` (default: 1024)
- Use fewer devices or smaller batch size
- Switch to GPT-OSS-20B if using 120B

**❌ "Device connection failed"**
- Check all Wormhole devices are properly connected
- Verify TT-NN installation and device drivers
- Try restarting the demo

**❌ Slow performance**
- Ensure devices are properly networked (fabric configuration)
- Check for thermal throttling
- Verify model files are on fast storage (not network)

### Getting Help
- Check logs for detailed error messages
- Monitor device memory usage during inference
- Try with smaller configurations first (fewer devices, shorter sequences)

---

## 📈 Monitoring & Metrics

The demo automatically reports:
- **Time to First Token (TTFT)**: How quickly the first response arrives
- **Tokens per Second**: Generation speed per user
- **Total Throughput**: System-wide token generation rate
- **Memory Usage**: Device memory consumption
- **Device Utilization**: How efficiently devices are used

### Sample Metrics Output
```
=== Performance Metrics ===
Average Time to First Token (TTFT): 180.5ms
Average decode speed: 35.2ms @ 28.4 tok/s/user (28.4 tok/s throughput)
Data parallel: 1, Global batch size: 1
```

---

## 🎉 What's Next?

### Experiment Ideas
1. **Creative Writing**: Try story prompts, poetry, scripts
2. **Code Generation**: Ask for programming help and examples
3. **Q&A**: Test knowledge across different domains
4. **Reasoning**: Logic puzzles and multi-step problems
5. **Conversation**: Multi-turn dialogue and chat

### Integration Options
- **REST API**: Wrap the demo in a web service
- **Batch Processing**: Process files with multiple prompts
- **Fine-tuning**: Adapt models for specific use cases
- **Multi-model**: Compare GPT-OSS-20B vs GPT-OSS-120B responses

### Performance Optimization
- **Larger Device Configurations**: Scale to 16, 32, or more devices
- **Custom Mesh Layouts**: Optimize device topology for your use case
- **Memory Optimization**: Implement custom caching strategies
- **Latency Tuning**: Optimize for fastest possible responses

---

## 💡 Tips for Best Results

### Prompt Engineering
- **Be specific**: "Write a 200-word summary of..." vs "Tell me about..."
- **Provide context**: Include relevant background information
- **Use examples**: Show the format you want in your prompt
- **Set constraints**: Specify length, style, or format requirements

### Performance Tips
- **Warm-up runs**: First inference is slower (compilation time)
- **Consistent lengths**: Similar sequence lengths perform better
- **Batch efficiently**: Group similar requests together
- **Monitor resources**: Watch memory and device utilization

---

**Ready to explore the power of Mixture of Experts language models? Start with the simple demo and scale up as needed!** 🚀
