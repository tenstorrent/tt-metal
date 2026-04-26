# CosyVoice Performance

## Scope

This demo tracks the public CosyVoice 300M bringup workload for:

- semantic token generation
- flow-based acoustic decoding
- HiFT vocoder synthesis

The current measured backend split is `tt_semantic_tt_flow_frontend_length_regulator_torch_decoder_reference_vocoder`.

The public performance command is:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/validate_tt.py \
  --suite performance \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output-json /tmp/cosy_voice_performance.json
```

The public quality command is:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/validate_tt.py \
  --suite quality \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output-json /tmp/cosy_voice_quality.json
```

## Public Gate

The target public workload should eventually prove:

- semantic token generation `>= 30 tokens/s`
- end-to-end `RTF < 0.5`

Only publish measured numbers from the accepted public workload and repo-local runtime.

## Report Fields

The public performance report should include:

- mode
- model id or local model path
- wall-clock seconds
- output audio seconds
- RTF
- semantic token count when available
- semantic tokens per second when available
- runtime verification fields

## Notes

- Keep README, CLI behavior, test thresholds, and measured artifacts aligned.
- Keep quality reports separate from semantic-parity and throughput reports.
- Do not publish full end-to-end TT performance claims until the diffusion decoder loop and HiFT vocoder are also on the TT path.
