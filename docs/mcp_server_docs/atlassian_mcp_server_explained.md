# How the Atlassian MCP Server Works

## What is the Atlassian MCP Server?

The Atlassian MCP server connects Claude to Tenstorrent's Atlassian cloud instance, giving access to both **Confluence** (wiki/documentation) and **Jira** (issue tracking). It exposes a large set of tools — over 30 — that map closely to the Atlassian REST API surface. Unlike a RAG-based server that synthesizes answers, this server acts as a **thin API proxy**: Claude calls a tool, the server translates it to an Atlassian REST API call, and the raw result comes back.

## Architecture

```
┌─────────────┐       ┌──────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Claude (LLM) │ ───── │  MCP Client  │ ───── │  Atlassian MCP   │ ───── │  Atlassian Cloud │
│             │       │  (in Claude  │       │  Server          │       │  (Confluence +   │
│             │       │   Code CLI)  │       │                  │       │   Jira REST API) │
└─────────────┘       └──────────────┘       └──────────────────┘       └──────────────────┘
```

1. **Atlassian MCP Server** — An external process that authenticates with the Atlassian cloud and exposes tools for reading/writing Confluence pages, searching Jira issues, managing comments, etc.
2. **MCP Client** — Built into Claude Code. Routes `mcp__atlassian__*` tool calls to the server.
3. **The LLM (Claude)** — Sees all Atlassian tools alongside built-in tools and decides when to call them.

## Authentication

Before any tools become available, the MCP server must authenticate with Atlassian. This happens via OAuth or API token flow. In practice, when you run `/mcp` in Claude Code and see `Authentication successful. Connected to atlassian.`, the server has obtained credentials and is ready to proxy requests.

Every API call requires a **Cloud ID** — a UUID that identifies the Atlassian site. This ID must be obtained first via `getAccessibleAtlassianResources` and then passed to every subsequent tool call.

## Core Concept: The Cloud ID

Almost every Atlassian tool requires a `cloudId` parameter. This is the identifier for the Atlassian cloud site (e.g., `b9d94484-5dbd-4ae2-b670-6f414aefb4cd` for `tenstorrent.atlassian.net`).

**Typical first call in any session:**

```json
{
  "name": "mcp__atlassian__getAccessibleAtlassianResources",
  "parameters": {}
}
```

This returns a list of accessible Atlassian sites with their Cloud IDs. The Cloud ID is then passed to all subsequent calls.

---

## Tools Overview

The Atlassian MCP server exposes tools in several categories. Below is a complete inventory.

### Identity & Discovery

| Tool | Purpose |
|---|---|
| `atlassianUserInfo` | Get current authenticated user info |
| `getAccessibleAtlassianResources` | Get Cloud IDs for accessible Atlassian sites |

### Confluence — Reading

| Tool | Purpose | Key Parameters |
|---|---|---|
| `getConfluencePage` | Get a single page by ID, including body content | `cloudId`, `pageId`, `contentFormat` (`markdown` or `adf`) |
| `getConfluenceSpaces` | List available spaces | `cloudId`, optional filters (`keys`, `labels`, `type`, `status`) |
| `getPagesInConfluenceSpace` | List pages within a space | `cloudId`, `spaceId`, optional `title` filter and `sort` |
| `getConfluencePageDescendants` | Get child pages of a given page | `cloudId`, `pageId`, optional `depth` |
| `getConfluencePageFooterComments` | Get footer comments on a page | `cloudId`, `pageId` |
| `getConfluencePageInlineComments` | Get inline (highlight) comments on a page | `cloudId`, `pageId`, optional `resolutionStatus` |
| `getConfluenceCommentChildren` | Get replies to a comment | `cloudId`, `commentId`, `commentType` |
| `searchConfluenceUsingCql` | Search Confluence content with CQL (Confluence Query Language) | `cloudId`, `cql` query string |

### Confluence — Writing

| Tool | Purpose |
|---|---|
| `createConfluencePage` | Create a new page (markdown or ADF) |
| `updateConfluencePage` | Update an existing page's content |
| `createConfluenceFooterComment` | Add a footer comment to a page |
| `createConfluenceInlineComment` | Add an inline comment on specific text |

### Jira — Reading

| Tool | Purpose | Key Parameters |
|---|---|---|
| `getJiraIssue` | Get full issue details by key (e.g., `PROJ-123`) | `cloudId`, `issueIdOrKey` |
| `searchJiraIssuesUsingJql` | Search issues with JQL (Jira Query Language) | `cloudId`, `jql`, optional `fields`, `maxResults` |
| `getVisibleJiraProjects` | List accessible Jira projects | `cloudId` |
| `getJiraProjectIssueTypesMetadata` | Get issue types for a project | `cloudId`, `projectIdOrKey` |
| `getJiraIssueTypeMetaWithFields` | Get field metadata for an issue type | `cloudId`, `projectIdOrKey`, `issueTypeId` |
| `getTransitionsForJiraIssue` | Get available status transitions | `cloudId`, `issueIdOrKey` |
| `getJiraIssueRemoteIssueLinks` | Get remote links on an issue | `cloudId`, `issueIdOrKey` |
| `lookupJiraAccountId` | Look up user account IDs by name | `cloudId`, `searchString` |
| `jiraRead` | Generic read operations (e.g., `getIssueLinkTypes`) | `cloudId`, `action` |

### Jira — Writing

| Tool | Purpose |
|---|---|
| `createJiraIssue` | Create a new issue |
| `editJiraIssue` | Update issue fields |
| `transitionJiraIssue` | Change issue status |
| `addCommentToJiraIssue` | Add a comment to an issue |
| `addWorklogToJiraIssue` | Log work time on an issue |
| `jiraWrite` | Generic write operations (e.g., `createIssueLink`) |

### Unified Search

| Tool | Purpose |
|---|---|
| `search` | Cross-product search (Rovo Search) across both Jira and Confluence |
| `fetch` | Fetch details of any Atlassian resource by its ARI (Atlassian Resource Identifier) |

---

## How Claude Accesses Confluence Data

### Typical Workflow: Finding and Reading a Page

Accessing Confluence content usually requires **multiple sequential tool calls** because each call returns only what was asked for — there is no synthesized "answer" like DeepWiki's `ask_question`.

**Step 1 — Search for the page:**

```json
{
  "name": "mcp__atlassian__search",
  "parameters": {
    "query": "Tensix ISA instruction set architecture"
  }
}
```

This returns a list of matching results with titles, snippets, page IDs, and URLs. The snippets are short and often sufficient to answer simple questions without further calls.

**Step 2 — Fetch the full page:**

```json
{
  "name": "mcp__atlassian__getConfluencePage",
  "parameters": {
    "cloudId": "b9d94484-5dbd-4ae2-b670-6f414aefb4cd",
    "pageId": "1613201604",
    "contentFormat": "markdown"
  }
}
```

This returns the page's full body content as markdown.

### Alternative: CQL Search

For structured searches, Confluence Query Language (CQL) provides more control:

```json
{
  "name": "mcp__atlassian__searchConfluenceUsingCql",
  "parameters": {
    "cloudId": "b9d94484-5dbd-4ae2-b670-6f414aefb4cd",
    "cql": "title ~ \"Tensix ISA\" AND type = page AND space = TA"
  }
}
```

### Navigation: Spaces → Pages → Children

You can also browse hierarchically:

```
getConfluenceSpaces → getPagesInConfluenceSpace → getConfluencePageDescendants → getConfluencePage
```

Each step narrows the scope but requires a separate tool call (and therefore a separate LLM round-trip).

---

## How Claude Accesses Jira Data

### Typical Workflow: Finding and Reading Issues

**Step 1 — Search for issues using JQL:**

```json
{
  "name": "mcp__atlassian__searchJiraIssuesUsingJql",
  "parameters": {
    "cloudId": "b9d94484-5dbd-4ae2-b670-6f414aefb4cd",
    "jql": "project = GHTTMETAL AND status = Open AND text ~ \"ISA\"",
    "fields": ["summary", "description", "status", "priority"],
    "maxResults": 10
  }
}
```

**Step 2 — Get full details of a specific issue:**

```json
{
  "name": "mcp__atlassian__getJiraIssue",
  "parameters": {
    "cloudId": "b9d94484-5dbd-4ae2-b670-6f414aefb4cd",
    "issueIdOrKey": "IPDOC-76"
  }
}
```

### Unified Search (Rovo)

The `search` tool queries across both Jira and Confluence in one call:

```json
{
  "name": "mcp__atlassian__search",
  "parameters": {
    "query": "Tensix ISA"
  }
}
```

This is often the best starting point because it returns results from both products without needing to decide which to query first.

---

## Context Cost Analysis

The Atlassian MCP server is **the most context-expensive** MCP integration. This is because it acts as a raw API proxy rather than a synthesis engine.

### Why Atlassian Calls Are Expensive

| Factor | Explanation |
|---|---|
| **Raw, unfiltered responses** | The server returns the full Atlassian API response — every field, every metadata object. A single Confluence page can be 50k-175k+ characters. |
| **Multi-step workflows** | Reading a Confluence page requires at minimum 2 calls (search → getPage), sometimes 3-4 for hierarchical navigation. Each call consumes context in both the request and response. |
| **No server-side summarization** | Unlike RAG-based servers, the Atlassian MCP returns verbatim content. A 100-page Confluence document comes back in full. |
| **Large Jira responses** | Jira issues carry extensive metadata (custom fields, changelog, transitions, etc.) even when only the summary is needed. |
| **Overflow to file** | When responses exceed ~25k tokens, Claude Code saves them to disk and provides a file path. Claude must then read the file with `Read`, `Grep`, or `Bash`, adding more round-trips and context. |

### Concrete Example: The ISA Query

When asked about the Tenstorrent ISA, the actual flow was:

1. **`search`** — Returned 20 results across Confluence and Jira (~3k tokens in context)
2. **`getConfluencePage` (ISA overview)** — Returned ~4k tokens of structured instruction listing
3. **`getConfluencePage` (SFPU ISA)** — Response was **175,548 characters**, overflowed to a file on disk
4. **`Grep` on overflow file** — Searching for key sections within the saved file
5. **`Bash` (python3 JSON extraction)** — Two calls to parse and extract sections from the JSON file

**Total: 6 tool calls and ~15-20k tokens of context consumed** to answer what was conceptually a single question.

### Mitigation Strategies

When using the Atlassian MCP server, Claude applies these strategies to reduce context cost:

1. **Start with `search` (Rovo)** — The unified search returns short snippets from both Jira and Confluence. Often the snippets alone answer the question, avoiding a full page fetch.

2. **Use `contentFormat: "markdown"`** — When fetching Confluence pages, markdown format is more compact than ADF (Atlassian Document Format), which is a verbose JSON structure.

3. **Handle overflows with targeted extraction** — When responses overflow to files, use `Grep` to find relevant sections rather than reading the entire file. Use `Bash` with a Python JSON parser to extract specific fields.

4. **Limit Jira `fields`** — When searching Jira issues, specify only the fields you need:
   ```json
   "fields": ["summary", "status", "priority"]
   ```
   Without this, every custom field on the issue is returned.

5. **Set `maxResults` conservatively** — Default Jira search can return up to 100 issues. Use `maxResults: 10` or less unless exhaustive results are needed.

6. **Prefer CQL/JQL over broad search** — Structured queries return fewer, more relevant results than keyword search.

---

## Tool Invocation Flow (Step by Step)

Here is the detailed flow when Claude calls an Atlassian tool:

### 1. Tool Discovery (session start)

When Claude Code starts and the Atlassian MCP server is configured, the client connects and discovers all 30+ tools. Each tool's schema (name, description, parameters) is injected into Claude's system prompt.

### 2. Authentication

When the user runs `/mcp`, the server performs OAuth authentication against `tenstorrent.atlassian.net`. On success, the server holds a valid token for subsequent API calls.

### 3. Tool Invocation (during conversation)

Claude emits a structured function call:

```json
{
  "name": "mcp__atlassian__getConfluencePage",
  "parameters": {
    "cloudId": "b9d94484-5dbd-4ae2-b670-6f414aefb4cd",
    "pageId": "1613201604",
    "contentFormat": "markdown"
  }
}
```

### 4. Request Routing (MCP Client)

Claude Code's MCP client strips the `mcp__atlassian__` prefix, identifies the `atlassian` server and the `getConfluencePage` tool, and forwards the request to the server process.

### 5. Server Processing (Atlassian MCP side)

The server:
- Maps the tool call to an Atlassian REST API endpoint (e.g., `GET /wiki/api/v2/pages/{pageId}?body-format=atlas_doc_format`)
- Attaches authentication headers
- Sends the HTTP request to `tenstorrent.atlassian.net`
- Returns the raw API response

### 6. Response Return

The response is passed back through the MCP client to Claude. If the response is small enough, it appears inline as a tool result. If it exceeds the token limit (~25k tokens), it overflows to a file on disk, and Claude receives a path to that file instead.

### 7. Overflow Handling

When overflow occurs, Claude must use secondary tools to access the content:

```
Tool result: "Error: result (175,548 characters) exceeds maximum allowed tokens.
Output has been saved to /home/user/.claude/projects/.../tool-results/mcp-atlassian-....txt"
```

Claude then uses `Read` (with `offset`/`limit`), `Grep`, or `Bash` (with `python3` or `jq`) to extract relevant portions.

---

## Naming Convention

All Atlassian tools follow the pattern `mcp__atlassian__<toolName>`:

- `mcp__` — MCP-provided tool prefix
- `atlassian` — server name
- `<toolName>` — the specific operation (e.g., `getConfluencePage`, `searchJiraIssuesUsingJql`)

Tool names use camelCase and generally follow the pattern `<verb><Product><Entity>` (e.g., `getConfluencePageFooterComments`, `createJiraIssue`).

---

## Practical Tips

### When to Use Which Tool

| Scenario | Recommended Tool | Why |
|---|---|---|
| Quick lookup across Jira + Confluence | `search` | Single call, short snippets, covers both products |
| Find a specific Confluence page | `searchConfluenceUsingCql` | CQL allows precise filtering by title, space, label |
| Read a known Confluence page | `getConfluencePage` | Direct access by page ID |
| Browse a Confluence space | `getConfluenceSpaces` → `getPagesInConfluenceSpace` | Hierarchical navigation |
| Find Jira issues | `searchJiraIssuesUsingJql` | JQL is powerful and precise |
| Get details of a known issue | `getJiraIssue` | Direct access by issue key |
| Create or update content | `create*` / `update*` / `edit*` tools | Write operations |

### Common Pitfalls

1. **Forgetting `cloudId`** — Nearly every tool requires it. Always call `getAccessibleAtlassianResources` first if you don't have it cached.
2. **Fetching large pages without need** — Check if search snippets answer the question before fetching full page content.
3. **Not specifying `fields` in Jira queries** — Without field filtering, Jira returns everything, including custom fields, changelogs, and rendered markup.
4. **Using `search` when CQL/JQL would be better** — The Rovo `search` tool is keyword-based. For structured queries (date ranges, specific projects, status filters), CQL/JQL is more precise and returns fewer irrelevant results.

---

## Summary

The Atlassian MCP server is a comprehensive but context-heavy integration. It exposes the full Atlassian REST API surface through 30+ tools, enabling Claude to read and write Confluence pages, search and manage Jira issues, and navigate the organizational knowledge base. The trade-off is that responses are raw and unfiltered — there is no server-side summarization or intelligence. This makes it essential to use targeted queries, field filtering, and overflow-handling strategies to keep context costs manageable.
