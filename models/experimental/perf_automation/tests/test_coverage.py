"""M2-review finding 2: playbook covers every op_class the parser can emit.

The parser's emittable op_class set (agent.opclass.EMITTABLE_OP_CLASSES) is the
closed vocabulary tracy_tool can produce. Run coverage_lint so a playbook gap
surfaces as a FAILURE here rather than as a silent zero-candidate route at
runtime (PLAN section 4.5 rule 6).
"""

from agent.opclass import EMITTABLE_OP_CLASSES
from agent.router import build_index, coverage_lint, route


def test_playbook_covers_emittable_keys():
    index = build_index()
    keys = [{"op_class": cls} for cls in sorted(EMITTABLE_OP_CLASSES)]
    uncovered = coverage_lint(index, keys)
    assert uncovered == [], f"playbook gap: op_classes with no section: {uncovered}"


def test_every_emittable_class_routes_to_at_least_one_section():
    index = build_index()
    for cls in sorted(EMITTABLE_OP_CLASSES):
        assert route(index, {"op_class": cls}), cls
