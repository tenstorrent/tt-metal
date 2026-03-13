# Evaluation: confluence-skill Repository

**Repository:** https://github.com/SpillwaveSolutions/confluence-skill
**Evaluated:** 2026-03-04
**Verdict:** Not adopted

## What It Is

A collection of Python CLI scripts and markdown reference docs for interacting with Confluence:

| Component | Purpose |
|-----------|---------|
| `scripts/confluence_auth.py` | Credential discovery from env vars, `.env` files, MCP config |
| `scripts/download_confluence.py` | Download pages to Markdown with attachment handling |
| `scripts/upload_confluence.py` | Upload Markdown to Confluence storage format |
| `scripts/upload_confluence_v2.py` | Updated upload script with image support |
| `scripts/convert_markdown_to_wiki.py` | Format conversion |
| `scripts/mermaid_renderer.py` | Mermaid diagram rendering for Confluence |
| `references/` | Wiki markup guide, storage format reference, troubleshooting |
| `SKILL.md` | Claude Code skill definition for registration |

## Security Review

No issues found. The scripts are straightforward REST API wrappers using `atlassian-python-api` and `requests`. No obfuscated code, no suspicious network calls, no data exfiltration. The auth module searches standard credential locations (environment variables, `.env` files, `~/.config/mcp/.mcp.json`).

## Reasons for Not Adopting

### 1. Redundant With Existing MCP Tools

The Atlassian MCP server already provides all necessary Confluence operations:

| Task | MCP Tool Available |
|------|--------------------|
| Search pages | `mcp__atlassian__search`, `mcp__atlassian__searchConfluenceUsingCql` |
| Read pages | `mcp__atlassian__getConfluencePage` |
| Create pages | `mcp__atlassian__createConfluencePage` |
| Update pages | `mcp__atlassian__updateConfluencePage` |
| Get child pages | `mcp__atlassian__getConfluencePageDescendants` |
| Comments | `mcp__atlassian__createConfluenceFooterComment`, `mcp__atlassian__getConfluencePageFooterComments` |
| Jira integration | Full suite of `mcp__atlassian__*Jira*` tools |

These tools are already authenticated and require no additional setup.

### 2. Dependency Conflicts

The scripts require packages not present in the tt-metal Python environment:

- `atlassian-python-api`
- `markdownify`
- `mistune`
- `PyYAML`
- `python-dotenv`
- `beautifulsoup4` (optional)

Installing these into `python_env/` risks version conflicts with tt-metal's pinned dependencies.

### 3. Separate Auth Mechanism

The scripts use REST API tokens discovered from `.env` files, while the MCP server already handles authentication transparently. Maintaining a second credential path adds complexity with no benefit.

### 4. Large Page Handling Already Solved

The `SKILL.md` claims the scripts are needed for pages exceeding MCP's size limits (10-20KB). In practice, when MCP returns an oversized page, the content is saved to a local file that can be searched with `Grep` or read in chunks with `Read`. This workflow already works — demonstrated by successfully retrieving the 175KB "Tensix SFPU Instruction Set Architecture" page and extracting the "Exception Flags" section from it using only MCP tools and `Grep`.
