# How MCP Servers Work (DeepWiki Example)

## What is MCP?

MCP (Model Context Protocol) is a standardized protocol that allows AI assistants like Claude to interact with external services through a defined interface. Think of it as a **plugin system** — it extends capabilities by giving access to tools hosted by external servers.

## Architecture

The architecture has three layers:

```
┌─────────────┐       ┌──────────────┐       ┌─────────────────┐
│ Claude (LLM) │ ───── │  MCP Client  │ ───── │  MCP Server     │
│             │       │  (in Claude  │       │  (e.g. DeepWiki)│
│             │       │   Code CLI)  │       │                 │
└─────────────┘       └──────────────┘       └─────────────────┘
```

1. **MCP Server** — An external process (local or remote) that exposes a set of **tools** and **resources** via the MCP protocol. DeepWiki is one such server.
2. **MCP Client** — Built into Claude Code (the CLI). It connects to configured MCP servers at startup, discovers their tools, and makes them available to the LLM.
3. **The LLM (Claude)** — Sees the MCP tools listed alongside all other tools (Bash, Read, Edit, etc.) and can call them like any other tool.

## How DeepWiki Specifically Works

DeepWiki is an MCP server that provides AI-powered documentation for GitHub repositories. When a session starts, Claude Code connects to the DeepWiki MCP server, and Claude receives a description of its tools.

### Tools Exposed by DeepWiki

DeepWiki exposes three tools, each serving a different level of detail. They form a natural hierarchy: structure gives you the map, contents gives you everything, and ask_question lets you query intelligently.

#### 1. `read_wiki_structure` — Get the Topic Map

This tool returns a structured table of contents for a repository's documentation. It lists all available topics and sections without returning their full content. Use it when you need to understand **what documentation exists** before diving deeper.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `repoName` | string | Yes | GitHub repository in `owner/repo` format |

**Example — Discovering what topics exist for tt-metal:**

```json
{
  "name": "mcp__deepwiki__read_wiki_structure",
  "parameters": {
    "repoName": "tenstorrent/tt-metal"
  }
}
```

**Typical response structure:**
```
- Overview
- Architecture
  - Tensix Core Architecture
  - Network on Chip (NoC)
  - Memory Hierarchy
- Programming Model
  - Kernels (Reader, Compute, Writer)
  - Circular Buffers
  - ...
- TTNN Operations
  - ...
```

**When Claude uses this:** Before asking a detailed question, Claude might call this first to see if a relevant topic even exists — similar to scanning a book's table of contents before reading a chapter.

---

#### 2. `read_wiki_contents` — Get Full Documentation

This tool returns the complete documentation content for a repository. It is the most comprehensive but also the heaviest call — it returns all topics with their full text.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `repoName` | string | Yes | GitHub repository in `owner/repo` format |

**Example — Getting full documentation for tt-metal:**

```json
{
  "name": "mcp__deepwiki__read_wiki_contents",
  "parameters": {
    "repoName": "tenstorrent/tt-metal"
  }
}
```

**Typical response:** A large markdown document containing all indexed documentation sections, including architecture descriptions, API references, code explanations, and usage guides.

**When Claude uses this:** Rarely in practice, because the response can be very large. Claude prefers `ask_question` for targeted queries or `read_wiki_structure` for navigation. This tool is most useful when a broad, comprehensive dump of all documentation is needed at once.

---

#### 3. `ask_question` — Ask a Targeted Question

This is the most frequently used and most powerful tool. It accepts a natural-language question about a repository and returns an AI-generated, context-grounded answer. Under the hood, DeepWiki likely uses RAG (Retrieval-Augmented Generation) to find relevant code and documentation, then synthesizes an answer.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `repoName` | string or string[] | Yes | One repo (`"owner/repo"`) or up to 10 repos (`["owner/repo1", "owner/repo2"]`) |
| `question` | string | Yes | The natural-language question to ask |

**Example 1 — Asking about a specific hardware concept:**

```json
{
  "name": "mcp__deepwiki__ask_question",
  "parameters": {
    "repoName": "tenstorrent/tt-metal",
    "question": "How do circular buffers synchronize data between reader and compute kernels on a Tensix core?"
  }
}
```

**Example 2 — Asking about an API pattern:**

```json
{
  "name": "mcp__deepwiki__ask_question",
  "parameters": {
    "repoName": "tenstorrent/tt-metal",
    "question": "What is the typical pattern for creating a new TTNN operation with custom compute kernels?"
  }
}
```

**Example 3 — Cross-repo question (comparing multiple repositories):**

```json
{
  "name": "mcp__deepwiki__ask_question",
  "parameters": {
    "repoName": ["tenstorrent/tt-metal", "pytorch/pytorch"],
    "question": "How does Tenstorrent's tile-based compute model differ from PyTorch's eager execution model?"
  }
}
```

**When Claude uses this:** This is the go-to tool when a user asks a question that local files (`METALIUM_GUIDE.md`, tech reports, source code) cannot fully answer. For example:

- User asks about an obscure hardware behavior → Claude calls `ask_question`
- User needs to understand how a subsystem works across many files → Claude calls `ask_question` instead of reading dozens of files
- User asks for conceptual explanations that benefit from synthesized documentation → Claude calls `ask_question`

---

### Tool Comparison at a Glance

| Tool | Granularity | Response Size | Best For |
|---|---|---|---|
| `read_wiki_structure` | High-level overview | Small | Discovering what topics exist |
| `read_wiki_contents` | Full detail, everything | Large | Comprehensive documentation dump |
| `ask_question` | Targeted, synthesized | Medium | Specific questions, most common use |

### Naming Convention

The tool names follow a pattern: `mcp__<server_name>__<tool_name>`. The `mcp__` prefix tells the system this is an MCP-provided tool, `deepwiki` identifies the server, and the rest is the tool name.

## How Claude Uses These Tools (Concrete Flow)

Here is what happens step-by-step when Claude calls `mcp__deepwiki__ask_question`:

### 1. Tool Discovery (at session start)

When Claude Code starts, it reads its configuration, sees that a DeepWiki MCP server is configured, connects to it, and asks: "What tools do you offer?" The server responds with JSON schemas describing each tool's name, description, and parameters. These get injected into Claude's system prompt as available functions.

### 2. Tool Invocation (during conversation)

When Claude decides to call the tool, it emits a structured function call like:

```json
{
  "name": "mcp__deepwiki__ask_question",
  "parameters": {
    "repoName": "tenstorrent/tt-metal",
    "question": "How do circular buffers work in the Tensix architecture?"
  }
}
```

### 3. Request Routing (MCP Client)

Claude Code's MCP client intercepts this call, strips the `mcp__deepwiki__` prefix to identify the server and tool, then forwards the request to the DeepWiki server using the MCP protocol (typically JSON-RPC over stdio or HTTP/SSE).

### 4. Server Processing (DeepWiki side)

The DeepWiki server receives the request, does its own processing — in this case, it likely:
- Looks up indexed/cached documentation for `tenstorrent/tt-metal`
- Runs a retrieval-augmented generation (RAG) pipeline against the repo's contents
- Returns a grounded answer

### 5. Response Return

The server sends back a response (text content), which Claude Code passes back to Claude as the tool result. Claude then incorporates that information into its reply to the user.

## How MCP Tools Differ From Built-in Tools

| Aspect | Built-in Tools (Read, Bash, etc.) | MCP Tools (DeepWiki, etc.) |
|---|---|---|
| **Execution** | Handled directly by Claude Code | Delegated to external server |
| **Availability** | Always present | Only if configured in settings |
| **Scope** | Local filesystem, shell | Anything the server implements |
| **Trust** | First-party | Depends on server configuration |

## Configuration

MCP servers are configured in Claude Code's settings (typically in `.claude/settings.json` or the global Claude Code config). A configuration entry specifies:

- **How to launch/connect** to the server (command + args for stdio-based servers, or a URL for HTTP-based servers)
- **Environment variables** the server might need (API keys, etc.)
- **Which tools to allow** (optional allowlists/blocklists)

## When Claude Uses DeepWiki in This Repo

Per the project instructions in `CLAUDE.md`, Claude is told to use DeepWiki proactively when local documentation (like `METALIUM_GUIDE.md` or the tech reports) is insufficient. For example, if a user asks about a hardware detail that is not covered locally, Claude would call:

```
mcp__deepwiki__ask_question(
  repoName: "tenstorrent/tt-metal",
  question: "your question here"
)
```

This gives Claude access to deeper, AI-indexed knowledge about the entire repository without needing to read every file directly.

## Summary

MCP is essentially a **tool-extension protocol**. Servers expose capabilities, clients discover and proxy them, and the LLM calls them as functions. DeepWiki is one example that turns a GitHub repository into a queryable knowledge base — but the same pattern applies to any MCP server (databases, APIs, internal tools, etc.).
