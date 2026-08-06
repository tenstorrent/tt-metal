"""mini-swe-agent model class for DiffusionGemma behind an OpenAI-compatible server.

This is the ONLY harness deviation in the frozen SWE-Bench Verified baseline, and it is a
transport-layer fix, not an accommodation of model output quality:

  The server runs with --reasoning-parser gemma4. On agentic turns the parser frequently
  classifies the WHOLE response as reasoning and leaves message.content empty, even though
  the model emitted a correctly formatted ```mswea_bash_command block. Any harness that
  parses actions out of content then sees zero actions. mini-swe-agent kills such instances
  with RepeatedFormatError in under a minute (8/8 in the first smoke), which is easily
  misread as the model failing the task.

  Fix: when content is blank, fall back to the reasoning text. The normal
  (thinking + answer) split is left untouched.

Everything else stays stock: the ```mswea_bash_command fence is parsed strictly, and no
attempt is made to repair malformed or degenerate output. A tolerant fence parser
(accepting mswea_bash_/msa_bash_command/```bash/bare fences, last-block-wins) WAS built and
REFUTED: replayed offline over 540 recorded format-error turns it recovered only 11.3%, and
the recovered "commands" were degenerate text (ls output pasted as a command, "the the the",
"mmmm......"). Executing those would manufacture a fake run. Do not reintroduce it.
"""

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


def _reasoning_of(msg) -> str:
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(msg, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    psf = getattr(msg, "provider_specific_fields", None) or {}
    if isinstance(psf, dict):
        for key in ("reasoning", "reasoning_content"):
            val = psf.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


class DGTextbasedModel(LitellmTextbasedModel):
    def _query(self, messages, **kwargs):
        response = super()._query(messages, **kwargs)
        try:
            msg = response.choices[0].message
            if not (getattr(msg, "content", None) or "").strip():
                reasoning = _reasoning_of(msg)
                if reasoning:
                    msg.content = reasoning
        except Exception:
            pass
        return response
