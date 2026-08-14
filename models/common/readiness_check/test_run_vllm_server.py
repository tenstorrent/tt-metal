from types import SimpleNamespace

from models.common.readiness_check.run_vllm_server import _stream_choice_is_token_event


def test_stream_choice_is_token_event_counts_empty_text_tokens():
    assert _stream_choice_is_token_event(SimpleNamespace(text="hello", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text=" ", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason=None))


def test_stream_choice_is_token_event_ignores_terminal_empty_chunks():
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="stop"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="length"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text=None, finish_reason=None))
