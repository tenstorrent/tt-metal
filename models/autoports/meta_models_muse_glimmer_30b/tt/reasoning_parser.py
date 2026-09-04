# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM reasoning parser for ``meta-models/Muse-Glimmer-30B``.

Muse Glimmer is a *channelled* model.  Its chat template ends an assistant turn
prompt at ``<|start|>assistant`` and the model itself writes the channel header,
so a single assistant turn looks like this on the wire::

    <|start|>assistant to=self<|message|>   ... analysis ...        <|eom|>
    <|start|>assistant to=user<|message|>   ... the actual answer ... <|eot|>

``<|start|>``, ``<|message|>``, ``<|eom|>`` and ``<|eot|>`` are special tokens,
so the detokenized text an OpenAI client sees is the two channels run together
with only their headers left as ordinary text::

    " to=self ... analysis ...assistant to=user ... the actual answer ..."

Without a reasoning parser vLLM hands that whole string back as
``choices[].message.content``.  That is lossless but wrong for any client that
expects ``content`` to be the model's reply: every eval harness, every
instruction-following check and every chat UI then reads the analysis channel as
part of the answer.  It is also non-conformant for a reasoning model --
``reasoning_content`` stays ``null`` even though the model produced reasoning.

This parser splits the channels: the ``self`` channel becomes
``reasoning_content`` and everything else (normally the ``user`` channel)
becomes ``content``.  When vLLM composes it with Muse Glimmer's ATEM tool
parser, it preserves a tool message's framing long enough for that parser to
identify and validate the call.  It is purely an API-layer reformat of text the
server already produced; it does not touch sampling, the generator, or
anything on device.

**The parser never removes information.**  It splits only a turn that actually
reached a visible channel.  A turn cut short before that -- ``max_tokens``
exhausted mid-analysis, or a ``stop`` string that matched inside the analysis
channel, both of which are ordinary for a model that always thinks first -- has
no visible channel to report, and reporting ``content=None`` for it would throw
away every token the model produced.  Such a turn is returned unsplit: exactly
the string an unparsed server would have returned.  So `content` is a string for
every response this server can produce, and enabling the parser can only ever
move the analysis of a *completed* turn out of `content` -- it can never empty
it.  (vLLM's own ``<think>``-style parsers do return ``content=None`` in this
case; that behaviour breaks any client that treats `content` as a string, which
includes tt-inference-server's own chat-completions parameter-conformance
suite.)

Enable it with::

    --reasoning-parser-plugin models/autoports/meta_models_muse_glimmer_30b/tt/reasoning_parser.py
    --reasoning-parser muse_glimmer

Every other reasoning model in the tt-inference-server catalog (gemma4, qwen3,
gpt-oss, minimax, glm) is released with an equivalent ``reasoning_parser``
setting; this is the Muse Glimmer equivalent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


#: The analysis channel's recipient.  Everything sent to ``self`` is reasoning;
#: every other recipient (``user``, or a tool namespace) is visible content.
REASONING_RECIPIENT = "self"

#: The visible answer channel's recipient.
CONTENT_RECIPIENT = "user"

# When the ATEM tool parser is active it forces ``skip_special_tokens=False``:
# the channel delimiters therefore survive detokenization and, unlike the
# legacy stripped representation below, make an arbitrary tool recipient
# unambiguous.  Keep this parser self-contained rather than importing the tool
# parser: each file is also a valid standalone vLLM parser plugin.
_FULL_HEADER_RE = re.compile(r"(?:<\|start\|>\s*assistant)?[^\S\n]*to=(?P<recipient>[A-Za-z0-9_.\-]+)<\|message\|>")
_FULL_END_RE = re.compile(r"<\|eom\|>|<\|eot\|>")

#: A channel header as it survives detokenization.  Either it opens the very
#: first channel -- the prompt already supplied ``<|start|>assistant``, so the
#: model's first emitted text is ``" to=<recipient>"`` -- or it opens a
#: subsequent one, where ``<|start|>`` is dropped and the literal word
#: ``assistant`` is left glued to the end of the previous channel.
#:
#: In the stripped form the recipient is matched against a closed set rather than a character class,
#: because ``<|message|>`` is a special token and disappears too: the body runs
#: straight into the recipient name (``to=userphotosynthesis is ...``), so there
#: is no delimiter to stop a greedy match at. Tool-enabled requests retain the
#: special tokens, so arbitrary tool recipients are parsed by ``_FULL_HEADER_RE``.
_RECIPIENTS = (REASONING_RECIPIENT, CONTENT_RECIPIENT)
_HEADER_RE = re.compile(r"(?:\A[ ]?|assistant[ ]?)to=(" + "|".join(_RECIPIENTS) + r")")

#: Every literal a channel header can appear as.  Used while streaming to hold
#: back a tail that has not yet decided whether it is a header or ordinary text.
#: The chat template always writes a space before ``to=`` (``<|start|>assistant
#: to=self<|message|>``), so only the spaced forms are listed; the regex keeps
#: the space optional for robustness, but treating a bare trailing ``"t"`` as a
#: possible header would stall every streamed word ending in one.
_HEADER_LITERALS = tuple(prefix + "to=" + recipient for recipient in _RECIPIENTS for prefix in ("assistant ", " "))

#: How many leading tokens to decode when deciding whether the model opened the
#: analysis channel.  The header is 3-5 tokens; 12 is slack for a tokenizer that
#: splits it differently.
_HEAD_TOKENS = 12


def pending_header_len(text: str) -> int:
    """Length of the trailing run of ``text`` that could still become a header.

    ``"...done.assistant to=us"`` has not yet decided whether it is prose or the
    opening of the reply channel, so a streaming caller must not emit its last
    16 characters yet.  A *complete* header is never pending.
    """
    best = 0
    for literal in _HEADER_LITERALS:
        for n in range(min(len(literal) - 1, len(text)), best, -1):
            if text.endswith(literal[:n]):
                best = n
                break
    return best


def split_channels(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(recipient, body)`` pairs, in order.

    Returns ``[]`` when the text carries no channel header at all, which is what
    a continuation-style or grammar-constrained generation looks like.
    """
    full_matches = list(_FULL_HEADER_RE.finditer(text))
    if full_matches:
        segments: list[tuple[str, str]] = []
        for i, match in enumerate(full_matches):
            next_header = full_matches[i + 1].start() if i + 1 < len(full_matches) else len(text)
            terminator = _FULL_END_RE.search(text, match.end(), next_header)
            end = terminator.start() if terminator is not None else next_header
            segments.append((match.group("recipient"), text[match.end() : end]))
        return segments

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return []
    segments: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((match.group(1), text[start:end]))
    return segments


def _after_full_reasoning_channel(text: str) -> str:
    """Raw text after a completed full-token ``self`` message.

    The raw suffix is intentional: vLLM hands this value to the tool parser on
    the reasoning-to-tool transition, and that parser needs ``to=<tool>`` plus
    ``<|message|>`` to distinguish executable ATEM from quoted markup.
    """
    for header in _FULL_HEADER_RE.finditer(text):
        if header.group("recipient") != REASONING_RECIPIENT:
            continue
        terminator = _FULL_END_RE.search(text, header.end())
        return text[terminator.end() :] if terminator is not None else ""
    return ""


def _has_tool_channel(text: str) -> bool:
    """Whether full-token framing contains a non-self, non-user recipient."""
    return any(
        match.group("recipient") not in (REASONING_RECIPIENT, CONTENT_RECIPIENT)
        for match in _FULL_HEADER_RE.finditer(text)
    )


def reached_visible_channel(text: str) -> bool:
    """Whether ``text`` contains a channel the user is meant to see.

    False for a turn with no header at all, and for one that opened the analysis
    channel and was cut off before closing it.  Both are returned unsplit.
    """
    return any(recipient != REASONING_RECIPIENT for recipient, _ in split_channels(text))


class MuseGlimmerReasoningParser(ReasoningParser):
    """Route Muse Glimmer's ``self`` channel to ``reasoning_content``."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if self.model_tokenizer is None:
            raise ValueError(
                "MuseGlimmerReasoningParser needs the model tokenizer to locate " "the <|eom|> channel terminator."
            )
        vocab = self.vocab
        self.eom_token_id = vocab.get("<|eom|>")
        self.eot_token_id = vocab.get("<|eot|>")
        if self.eom_token_id is None:
            raise RuntimeError(
                "MuseGlimmerReasoningParser could not find <|eom|> in the "
                "tokenizer vocabulary; this parser only fits Muse Glimmer's "
                "channelled chat format."
            )
        # Cache the last (input_ids-prefix -> opened-analysis-channel) answer.
        # is_reasoning_end is called once per decode step by the structured
        # output engine, and the answer only depends on the first few tokens.
        self._head_cache: tuple[tuple[int, ...], bool] | None = None
        # Streaming bookkeeping: how much of each channel this request has
        # already been handed to the client.  vLLM builds one parser per
        # streaming request, so this is per-request state.
        self._emitted_reasoning = 0
        self._emitted_content = 0
        self._preserve_tool_frames = False

    def adjust_request(self, request):
        """Remember whether downstream ATEM parsing needs raw channel frames."""
        tools = getattr(request, "tools", None)
        self._preserve_tool_frames = bool(tools) and getattr(request, "tool_choice", None) != "none"
        return request

    # -- channel-state helpers -------------------------------------------------

    def _opened_analysis_channel(self, input_ids: Sequence[int]) -> bool:
        """Whether the generation's first channel header addressed ``self``.

        An empty or too-short generation counts as *not* opened: a caller asking
        before the header exists should not be told the model is reasoning.
        """
        head = tuple(input_ids[:_HEAD_TOKENS])
        if not head:
            return False
        if self._head_cache is not None and self._head_cache[0] == head:
            return self._head_cache[1]
        text = self.model_tokenizer.decode(list(head), skip_special_tokens=True)
        match = _HEADER_RE.search(text)
        opened = bool(match) and match.group(1) == REASONING_RECIPIENT
        self._head_cache = (head, opened)
        return opened

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """True once the analysis channel is closed -- or was never opened.

        The analysis channel is terminated by ``<|eom|>``.  When the model never
        opened one (a grammar-constrained generation, or a direct reply), there
        is no reasoning to wait for and the answer is True from the first step,
        which is what keeps structured output applying its grammar immediately.
        """
        if not self._opened_analysis_channel(input_ids):
            return True
        return self.eom_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """The token ids after the analysis channel closed."""
        if not self._opened_analysis_channel(input_ids):
            return list(input_ids)
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.eom_token_id:
                return list(input_ids[i + 1 :])
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._opened_analysis_channel(token_ids):
            return 0
        for i, token_id in enumerate(token_ids):
            if token_id == self.eom_token_id:
                return i + 1
        return len(token_ids)

    # -- text extraction -------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not reached_visible_channel(model_output):
            # Either no channel header at all, or the turn was cut off inside
            # the analysis channel. Nothing to route; hand the text back
            # unchanged rather than inventing a split or dropping the output.
            return None, model_output
        reasoning = self.channel_text(model_output, True)
        # DelegatingParser invokes reasoning extraction before tool extraction.
        # Preserve the original framed text only when a tool channel is really
        # present; the ATEM parser then removes self/user framing and returns the
        # structured call. A tools-enabled request that answers directly keeps
        # the normal clean user content path.
        if self._preserve_tool_frames and (
            _has_tool_channel(model_output)
            or (_FULL_HEADER_RE.search(model_output) is None and "<atem:invoke" in model_output)
        ):
            return (reasoning or None), model_output
        content = self.channel_text(model_output, False)
        return (reasoning or None), content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Emit the delta into whichever channel it now belongs to.

        The split is recomputed from ``current_text`` each step rather than from
        ``delta_text``, so a delta that straddles a channel boundary is divided
        rather than misfiled.  A trailing run that could still grow into a
        channel header is held back (see :func:`pending_header_len`) so the
        client never sees half of ``assistant to=user`` as reply text; it is
        released on the step that resolves it.

        Streaming cannot offer :meth:`extract_reasoning`'s unsplit-on-truncation
        guarantee: a delta has to be labelled when it is emitted, and whether a
        visible channel ever arrives is not known until the turn ends.  A turn
        cut off inside the analysis channel therefore streams as reasoning
        deltas with no content, which is what every other vLLM reasoning parser
        does.  Every eval, benchmark and conformance path in this release runs
        non-streaming.
        """
        if not delta_text:
            return None

        reasoning_all = self.channel_text(current_text, True)
        if self._preserve_tool_frames and _FULL_HEADER_RE.search(current_text):
            content_all = _after_full_reasoning_channel(current_text)
        else:
            content_all = self.channel_text(current_text, False)

        hold = pending_header_len(current_text)
        if hold:
            segments = split_channels(current_text)
            open_is_reasoning = bool(segments) and (segments[-1][0] == REASONING_RECIPIENT)
            if open_is_reasoning:
                reasoning_all = reasoning_all[: max(0, len(reasoning_all) - hold)]
            else:
                content_all = content_all[: max(0, len(content_all) - hold)]

        reasoning_delta = reasoning_all[self._emitted_reasoning :]
        content_delta = content_all[self._emitted_content :]
        self._emitted_reasoning += len(reasoning_delta)
        self._emitted_content += len(content_delta)

        if not reasoning_delta and not content_delta:
            # Nothing resolved this step: header characters, or held-back text.
            return None
        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )

    @staticmethod
    def channel_text(text: str, want_reasoning: bool) -> str:
        """One side of the channel split: the analysis channel, or everything else.

        With no channel header at all the whole string is content, matching
        :meth:`extract_reasoning`.
        """
        segments = split_channels(text)
        if not segments:
            return "" if want_reasoning else text
        return "".join(body for recipient, body in segments if (recipient == REASONING_RECIPIENT) is want_reasoning)


# vLLM 0.24's decorator form records a lazy module import.  Parser plugins are
# loaded from a file path, however, so the temporary module name used by the
# loader is not necessarily importable later.  Register the already-defined
# class eagerly to make both the plugin verifier and the live parser lookup
# independent of that implementation detail.
ReasoningParserManager.register_module(
    name="muse_glimmer",
    force=True,
    module=MuseGlimmerReasoningParser,
)


def iter_channel_recipients(text: str) -> Iterable[str]:
    """The channel recipients ``text`` addresses, in order.  Test/debug helper."""
    return (recipient for recipient, _ in split_channels(text))
