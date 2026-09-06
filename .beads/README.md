# Beads - AI-Native Issue Tracking

Welcome to Beads! This repository uses **Beads** for issue tracking - a modern, AI-native tool designed to live directly in your codebase alongside your code.

## What is Beads?

Beads is issue tracking that lives in your repo, making it perfect for AI coding agents and developers who want their issues close to their code. No web UI required - everything works through the CLI and integrates seamlessly with git.

**Learn more:** [github.com/steveyegge/beads](https://github.com/steveyegge/beads)

## Quick Start

### Essential Commands

```bash
# Create new issues
bd create "Add user authentication"

# View all issues
bd list

# View issue details
bd show <issue-id>

# Update issue status
bd update <issue-id> --claim
bd update <issue-id> --status done

# Sync with Dolt remote
bd dolt push
```

### Working with Issues

Issues in Beads are:
- **Git-native**: Stored in Dolt database with version control and branching
- **AI-friendly**: CLI-first design works perfectly with AI coding agents
- **Branch-aware**: Issues can follow your branch workflow
- **Sync-ready**: Uses Dolt remotes for backup and team sharing

## Why Beads?

✨ **AI-Native Design**
- Built specifically for AI-assisted development workflows
- CLI-first interface works seamlessly with AI coding agents
- No context switching to web UIs

🚀 **Developer Focused**
- Issues live in your repo, right next to your code
- Works offline, syncs when you push
- Fast, lightweight, and stays out of your way

🔧 **Git Integration**
- Dolt-native sync via bd dolt push / bd dolt pull
- Branch-aware issue tracking
- Dolt-native three-way merge resolution

## Get Started with Beads

Try Beads in your own projects:

```bash
# Install Beads
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# Initialize in your repo
bd init

# Create your first issue
bd create "Try out Beads"
```

## Learn More

- **Documentation**: [github.com/steveyegge/beads/docs](https://github.com/steveyegge/beads/tree/main/docs)
- **Quick Start Guide**: Run `bd quickstart`
- **Examples**: [github.com/steveyegge/beads/examples](https://github.com/steveyegge/beads/tree/main/examples)

---

*Beads: Issue tracking that moves at the speed of thought* ⚡

## Resume this repository in another checkout

Task data is published separately from code through the Dolt Git remote
`git+https://github.com/tenstorrent/tt-metal.git` (`refs/dolt/data`). A normal
Git branch push does not publish the task database, and checking out a code
branch does not restore it automatically. The task database is shared across
code branches, rather than being a per-code-branch snapshot.

With Beads 1.2.2 (the version used to verify recovery), initialize a fresh clone:

```sh
bd init --remote git+https://github.com/tenstorrent/tt-metal.git --skip-hooks --skip-agents --non-interactive
bd ready --json
bd list --status=in_progress --json
```

For a checkout that already has a Beads database, use `bd dolt pull` instead of
initializing it again. Preserve local task edits and resolve any sync conflicts.
To publish subsequent task updates, use `bd dolt push` as well as the normal code
branch push. Both operations require access to this GitHub repository.

Local Git stashes and ignored profiling output do not travel with a branch push.
Keep experiment archives separately; task descriptions and notes contain the
scope, outcomes, and references needed to resume the work.
