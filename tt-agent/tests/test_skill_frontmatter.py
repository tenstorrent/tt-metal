"""Validates that all SKILL.md files in tt-agent/skills/ have correct YAML frontmatter."""
import os
import re
import yaml
import pytest
from pathlib import Path

SKILLS_ROOT = Path(__file__).parent.parent / "skills"
KNOWLEDGE_ROOT = Path(__file__).parent.parent / "knowledge"
RECIPES_ROOT = KNOWLEDGE_ROOT / "recipes"


def find_skill_files():
    return list(SKILLS_ROOT.rglob("SKILL.md"))


def parse_frontmatter(skill_file):
    content = skill_file.read_text()
    end = content.find("\n---\n", 4)
    return yaml.safe_load(content[4:end])


def test_skill_files_exist():
    """At least one SKILL.md must exist."""
    files = find_skill_files()
    assert len(files) > 0, f"No SKILL.md files found under {SKILLS_ROOT}"


@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_has_valid_frontmatter(skill_file):
    """Every SKILL.md must have valid YAML frontmatter with name and description."""
    content = skill_file.read_text()
    assert content.startswith("---\n"), f"{skill_file}: must start with YAML frontmatter (---)"

    end = content.find("\n---\n", 3)
    assert end != -1, f"{skill_file}: frontmatter block not closed"

    frontmatter_str = content[4:end]
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        pytest.fail(f"{skill_file}: invalid YAML frontmatter: {e}")

    assert "name" in frontmatter, f"{skill_file}: missing 'name' field"
    assert "description" in frontmatter, f"{skill_file}: missing 'description' field"
    assert (
        isinstance(frontmatter["name"], str) and frontmatter["name"].strip()
    ), f"{skill_file}: 'name' must be a non-empty string"
    assert (
        isinstance(frontmatter["description"], str) and frontmatter["description"].strip()
    ), f"{skill_file}: 'description' must be a non-empty string"


VALID_LAYERS = {"orchestration", "workflow", "tool", "meta"}


@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_has_valid_layer(skill_file):
    """Skill layer must be declared in metadata.layer with a valid value."""
    content = skill_file.read_text()
    end = content.find("\n---\n", 4)
    frontmatter = yaml.safe_load(content[4:end])
    metadata = frontmatter.get("metadata", {})
    assert "layer" in metadata, f"{skill_file}: missing 'metadata.layer' — use metadata: {{ layer: <value> }}"
    assert (
        metadata["layer"] in VALID_LAYERS
    ), f"{skill_file}: metadata.layer '{metadata['layer']}' not in {VALID_LAYERS}"


@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_name_matches_directory(skill_file):
    """Skill name in frontmatter must match the parent directory name."""
    content = skill_file.read_text()
    end = content.find("\n---\n", 4)
    frontmatter = yaml.safe_load(content[4:end])
    dir_name = skill_file.parent.name
    assert (
        frontmatter["name"] == dir_name
    ), f"{skill_file}: name '{frontmatter['name']}' must match directory '{dir_name}'"


def find_workflow_skills():
    """Find all SKILL.md files with metadata.layer == 'workflow'."""
    results = []
    for skill_file in find_skill_files():
        frontmatter = parse_frontmatter(skill_file)
        if frontmatter.get("metadata", {}).get("layer") == "workflow":
            results.append(skill_file)
    return results


# Only run phase table tests if workflow skills exist
_workflow_skills = find_workflow_skills()


@pytest.mark.parametrize(
    "skill_file", _workflow_skills if _workflow_skills else [pytest.param(None, marks=pytest.mark.skip)]
)
def test_workflow_skill_has_phase_table(skill_file):
    """Workflow skills must have a phase table with Loads and Produces columns."""
    if skill_file is None:
        return
    content = skill_file.read_text()
    assert re.search(
        r"\|\s*Phase\s*\|", content
    ), f"{skill_file}: workflow skill must have a phase table (| Phase | Loads | Produces |)"


def _extract_loads_paths(content):
    """Extract file paths from Loads column in phase tables, ignoring <placeholder> paths."""
    paths = []
    for match in re.finditer(r"`([^`]+)`", content):
        path = match.group(1)
        # Skip paths with any <placeholder> — resolved at runtime (e.g. <repo>, <topic>)
        if re.search(r"<[^>]+>", path):
            continue
        # Only match paths that look like file references (contain / and end with .md)
        if "/" in path and path.endswith(".md"):
            paths.append(path)
    return paths


@pytest.mark.parametrize(
    "skill_file", _workflow_skills if _workflow_skills else [pytest.param(None, marks=pytest.mark.skip)]
)
def test_workflow_loads_files_exist(skill_file):
    """Every file referenced in a workflow skill's Loads column must exist on disk."""
    if skill_file is None:
        return
    content = skill_file.read_text()
    # Extract paths from the phase table section only
    phase_section = ""
    in_table = False
    for line in content.split("\n"):
        if re.match(r"\|\s*Phase\s*\|", line):
            in_table = True
        elif in_table:
            if line.startswith("|"):
                phase_section += line + "\n"
            else:
                in_table = False

    agent_root = Path(__file__).parent.parent
    skill_dir = skill_file.parent
    missing = []
    for path in _extract_loads_paths(phase_section):
        # Try relative to skill dir first, then agent root
        if not (skill_dir / path).exists() and not (agent_root / path).exists():
            missing.append(path)

    assert not missing, f"{skill_file}: Loads references missing files: {missing}"


# --- code-review skill structural checks ---

CODE_REVIEW_DIR = SKILLS_ROOT / "code-review"
REVIEWER_REQUIRED_SECTIONS = [
    "## Mission",
    "## Base Checklist",
    "## TT Checklist",
    "## Severity Definitions",
]


def find_reviewer_files():
    reviewers = CODE_REVIEW_DIR / "reviewers"
    if not reviewers.exists():
        return []
    return sorted(reviewers.glob("*.md"))


_reviewer_files = find_reviewer_files()


@pytest.mark.parametrize(
    "reviewer_file",
    _reviewer_files if _reviewer_files else [pytest.param(None, marks=pytest.mark.skip)],
)
def test_reviewer_file_has_required_sections(reviewer_file):
    """Each code-review reviewer file must contain the required section headers."""
    if reviewer_file is None:
        return
    content = reviewer_file.read_text()
    missing = [s for s in REVIEWER_REQUIRED_SECTIONS if s not in content]
    assert not missing, f"{reviewer_file}: missing required sections: {missing}"


# --- cross-reference check: orchestrator dispatch table lists every skill ---

ORCHESTRATOR_SKILL = SKILLS_ROOT / "orchestrator" / "SKILL.md"
# Skills intentionally excluded from the orchestrator dispatch table.
# - orchestrator: does not dispatch to itself
# - skill-creator: dev-facing, invoked directly when building new skills
ORCHESTRATOR_EXEMPT = {"orchestrator", "skill-creator"}


@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_referenced_in_orchestrator(skill_file):
    """Every skill (except the orchestrator itself) must be named in the dispatch table."""
    if not ORCHESTRATOR_SKILL.exists():
        pytest.skip("orchestrator SKILL.md not present")
    name = skill_file.parent.name
    if name in ORCHESTRATOR_EXEMPT:
        return
    content = ORCHESTRATOR_SKILL.read_text()
    # Accept either /tt:<name> or plain <name> mentions
    assert f"tt:{name}" in content or f"`{name}`" in content or f"/{name}" in content, (
        f"{skill_file}: skill '{name}' not referenced in orchestrator dispatch table " f"({ORCHESTRATOR_SKILL})"
    )


def find_recipe_dirs():
    """Find all recipe directories under knowledge/recipes/."""
    if not RECIPES_ROOT.exists():
        return []
    return [d for d in RECIPES_ROOT.iterdir() if d.is_dir()]


_recipe_dirs = find_recipe_dirs()


@pytest.mark.parametrize("recipe_dir", _recipe_dirs if _recipe_dirs else [pytest.param(None, marks=pytest.mark.skip)])
def test_recipe_has_index(recipe_dir):
    """Every recipe directory must have an index.md."""
    if recipe_dir is None:
        return
    index = recipe_dir / "index.md"
    assert index.exists(), f"{recipe_dir}: recipe directory must have an index.md"
