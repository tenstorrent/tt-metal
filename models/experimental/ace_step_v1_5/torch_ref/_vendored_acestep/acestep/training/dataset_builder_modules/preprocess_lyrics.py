import torch


def encode_lyrics(text_encoder, text_tokenizer, lyrics: str, device, dtype):
    """Encode lyrics into hidden states."""
    lyric_inputs = text_tokenizer(
        lyrics,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    lyric_input_ids = lyric_inputs.input_ids.to(device)
    lyric_attention_mask = lyric_inputs.attention_mask.to(device).to(dtype)

    # Align tensor residency to the actual text encoder device to avoid
    # CPU/CUDA mismatch in embedding/index_select calls.
    text_dev = next(text_encoder.parameters()).device
    if lyric_input_ids.device != text_dev:
        lyric_input_ids = lyric_input_ids.to(text_dev)
        lyric_attention_mask = lyric_attention_mask.to(text_dev)

    with torch.no_grad():
        lyric_hidden_states = text_encoder.embed_tokens(lyric_input_ids).to(dtype)

    return lyric_hidden_states, lyric_attention_mask
