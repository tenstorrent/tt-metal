"""A registry entry must name a folder that exists, or be a generic backend.

Twelve entries pointed at folders absent from the checkout. Nothing caught it:
routing matched them, bring-up produced nothing, and the failure surfaced later as
"discovery produced no module tree" -- a message about the model, for a problem in
the registry. FLUX.2-klein's text_encoder lost a whole component that way.

Two rules, and this file pins both:
  * a `template` backend copies its demo folder, so that folder must exist;
  * a `generic` backend WRITES models/demos/<family>/<model>/ per bring-up and has
    no source folder, so its demo_path names an output location and is exempt.
"""

from __future__ import annotations

import os

from scripts.tt_hw_planner.family_backends import all_backends, demo_path_exists
from scripts.tt_hw_planner.registry_sync import prunable_backends

# Entries deliberately registered before their demo has landed. Every name here is
# dead weight that routing skips; the list may only SHRINK. Adding to it means
# accepting a route that cannot work, so prefer landing the demo or deleting the
# entry instead.
KNOWN_ABSENT_DEMOS = {
    # Pinned by test_xtts_v2_is_in_tts_bucket and the only TTS backend registered,
    # so it cannot be removed without also deciding what TTS models should route to.
    "XTTS-v2 (multilingual TTS)",
}


def test_no_new_template_backend_points_at_a_missing_folder() -> None:
    dead = {name for name, _ in prunable_backends()}
    unexpected = dead - KNOWN_ABSENT_DEMOS
    assert not unexpected, (
        "these template backends name a folder that is not in this checkout, so they "
        "can only produce an empty bring-up -- land the demo, repoint the entry, or "
        f"delete it: {sorted(unexpected)}"
    )


def test_known_absent_list_has_not_gone_stale() -> None:
    """If a listed demo has landed, take it off the list rather than leaving cover
    for the next one."""
    dead = {name for name, _ in prunable_backends()}
    landed = KNOWN_ABSENT_DEMOS - dead
    assert not landed, f"demo now exists — remove from KNOWN_ABSENT_DEMOS: {sorted(landed)}"


def test_generic_backends_are_exempt_from_the_folder_rule() -> None:
    """A generic backend has no template to copy; its path is where it writes."""
    generic = [b for b in all_backends() if (b.routing_mode or "") == "generic"]
    assert generic, "expected at least one generic catch-all backend"
    reported = {name for name, _ in prunable_backends()}
    for b in generic:
        assert b.name not in reported, f"generic backend {b.name!r} must not be reported as prunable"


def test_every_routable_backend_either_exists_or_is_generic() -> None:
    for b in all_backends():
        if (b.routing_mode or "") == "generic" or b.name in KNOWN_ABSENT_DEMOS:
            continue
        assert demo_path_exists(b), f"{b.name}: demo_path {b.demo_path!r} does not exist"


def test_demo_path_exists_reads_the_filesystem_not_the_string() -> None:
    """Guard the helper itself: it must not be fooled by a plausible-looking path."""

    class _B:
        demo_path = os.path.join("definitely", "not", "a", "real", "path")
        routing_mode = "template"
        name = "synthetic"

    assert demo_path_exists(_B()) is False
