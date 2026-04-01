# Inworld TTS Bringup Log

## Session 1 - 2026-04-01

### Status: Phase 1 Architecture Mapping - COMPLETE

### Work Done
- Cloned and analyzed https://github.com/inworld-ai/tts
- Identified model as SpeechLM-based TTS: fine-tuned LLaMA + xcodec2-compatible codec
- Created complete component inventory (10 components)
- Mapped all components to existing TTNN reference implementations
- Created ARCHITECTURE.md with full weight mapping and implementation order

### Key Findings
- LLaMA LLM backbone can be reused from `models/demos/llama3_70b_galaxy/tt/`
- Codec uses FSQ (Finite Scalar Quantization), NOT RVQ -- single codebook with 65,536 entries
- Codec decoder uses VocosBackbone: 12 transformer layers (bidirectional, non-causal) + ISTFT
- VocosBackbone attention is MHA (not GQA), with RoPE, and bidirectional (is_causal=False)
- MLP is simple SiLU (not SwiGLU) -- no gate projection
- ISTFT head does signal processing (FFT) -- may need PyTorch fallback

### Implementation Priority
1. Codec Decoder (VocosBackbone + ISTFTHead) -- novel, needed for audio output
2. LLaMA LLM -- reuse existing with expanded vocab
3. Codec Encoder -- needed for voice cloning prompts

### Next Steps
- Phase 2: Create PyTorch reference implementation
  - Verify official package works end-to-end
  - Create standalone reference modules for codec decoder
  - Generate golden outputs

### PCC: N/A (no implementation yet)
### Block Hash: N/A
