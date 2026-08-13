from types import SimpleNamespace

from models.common.readiness_check.run_vllm_server import _stream_choice_is_token_event, _tokenizer_load_kwargs


def test_stream_choice_is_token_event_counts_empty_text_tokens():
    assert _stream_choice_is_token_event(SimpleNamespace(text="hello", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text=" ", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason=None))


def test_stream_choice_is_token_event_ignores_terminal_empty_chunks():
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="stop"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="length"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text=None, finish_reason=None))


def test_tokenizer_compatibility_kwargs_are_checkpoint_scoped():
    assert _tokenizer_load_kwargs("mistralai/Mistral-Small-24B-Instruct-2501") == {"fix_mistral_regex": True}
    assert _tokenizer_load_kwargs("meta-llama/Llama-3.1-8B") == {}
