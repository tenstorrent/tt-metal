# The `/context` Command in Claude Code

## What It Does

The `/context` command is a built-in Claude Code CLI command that displays a **colored grid visualizing your current context window usage**. It shows exactly how much of your available token budget is being consumed by different parts of your conversation and state.

It cannot be invoked programmatically from within a conversation (like `/help` or `/clear`, it is handled by the CLI itself, not as a skill).

## Grid Structure

Each colored block in the grid represents a proportion of your total context window (typically 200k tokens). The categories include:

| Section | Represents |
|---|---|
| **System prompts** | CLAUDE.md files, tool definitions, MCP server configs |
| **Conversation history** | Your messages and Claude's responses |
| **Tool results** | Output from Read, Bash, Grep, etc. |
| **Available space** | Remaining unused context |

## Concrete Example

Consider a session where you:

1. Have two CLAUDE.md files loaded (project-level and nested)
2. Have MCP server tool definitions (e.g., Atlassian with ~15 tools)
3. Fetched a large Confluence page (171K chars)
4. Ran several Python extraction scripts against the fetched content
5. Exchanged a few messages with Claude

Running `/context` in this session would show:

- A **large chunk** for system prompts — the CLAUDE.md files, MCP tool schemas, and built-in tool definitions form a substantial base cost.
- A **significant block** for tool results — the Confluence page fetch and Python script outputs are all stored in context.
- A **moderate block** for conversation — your messages and Claude's responses.
- The **remaining empty space** — how much room is left before context is exhausted.

## Why It Is Useful

- See when you are approaching the limit and need to `/compact` or `/clear`.
- Understand what is consuming the most space (e.g., a large file read vs. conversation).
- Plan whether you have room for more tool-heavy operations.

## Related Commands

| Command | Purpose |
|---|---|
| `/compact [instructions]` | Compress conversation while preserving key information |
| `/clear` | Start fresh with a new context window |
| `/cost` | See detailed token usage statistics |
