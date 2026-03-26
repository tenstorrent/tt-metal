"""Validates that all SKILL.md files in tt-agent/skills/ have correct YAML frontmatter."""
import os
import yaml
import pytest
from pathlib import Path

SKILLS_ROOT = Path(__file__).parent.parent / "skills"


def find_skill_files():
    return list(SKILLS_ROOT.rglob("SKILL.md"))


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
