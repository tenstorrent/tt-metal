# Atlassian MCP Server: API, Context Cost, and Usage Strategy

## What Is the Atlassian MCP Server?

MCP (Model Context Protocol) is a protocol that allows an LLM-based agent (Claude Code in this case) to call external tools provided by a separate server process. The Atlassian MCP server is a plugin that exposes Confluence and Jira APIs as callable tools within the Claude Code session.

When Claude Code starts, it connects to the configured MCP server. The server advertises a set of tools (functions), each with a JSON schema describing its parameters. Claude Code can then invoke these tools by name during a conversation, just like its built-in tools (Read, Write, Bash, etc.). The MCP server handles authentication, makes the actual HTTP requests to the Atlassian Cloud REST API, and returns the results.

---

## Available Atlassian MCP Tools

### Discovery and Navigation

| Tool | Purpose | Parameters |
|---|---|---|
| `atlassianUserInfo` | Get the authenticated user's identity | None |
| `getAccessibleAtlassianResources` | List Atlassian cloud sites and their cloud IDs | None |
| `getConfluenceSpaces` | List spaces in a Confluence site | `cloudId`, optional filters (`type`, `status`, `keys`, `labels`, `limit`) |
| `getPagesInConfluenceSpace` | List pages within a specific space | `cloudId`, `spaceId`, optional `title`, `sort`, `limit`, `cursor` |
| `getConfluencePageDescendants` | Get child pages of a given page | `cloudId`, `pageId`, optional `depth`, `limit`, `cursor` |

### Content Retrieval

| Tool | Purpose | Parameters |
|---|---|---|
| `getConfluencePage` | Fetch a full page including body content | `cloudId`, `pageId`, `contentFormat` (`markdown` or `adf`) |
| `fetch` | Fetch a Jira issue or Confluence page by ARI | `id` (Atlassian Resource Identifier) |
| `search` | Rovo Search across Jira and Confluence | `query` (natural language) |
| `searchConfluenceUsingCql` | Search Confluence with CQL query syntax | `cloudId`, `cql`, optional `limit`, `cursor`, `expand` |

### Comments

| Tool | Purpose | Parameters |
|---|---|---|
| `getConfluencePageFooterComments` | Get footer comments on a page | `cloudId`, `pageId`, optional `sort`, `limit`, `cursor` |
| `getConfluencePageInlineComments` | Get inline comments on a page | `cloudId`, `pageId`, optional `resolutionStatus`, `sort`, `limit` |
| `getConfluenceCommentChildren` | Get replies to a specific comment | `cloudId`, `commentId`, `commentType` (`footer` or `inline`) |

### Content Creation and Modification

| Tool | Purpose | Parameters |
|---|---|---|
| `createConfluencePage` | Create a new Confluence page | `cloudId`, `spaceId`, `body`, optional `title`, `parentId`, `contentFormat` |
| `updateConfluencePage` | Update an existing Confluence page | `cloudId`, `pageId`, `body`, optional `title`, `status`, `versionMessage` |
| `createConfluenceFooterComment` | Add a footer comment to a page | `cloudId`, `body`, optional `pageId`, `parentCommentId` |
| `createConfluenceInlineComment` | Add an inline comment highlighting specific text | `cloudId`, `body`, optional `pageId`, `inlineCommentProperties` (text selection) |

---

## How a Tool Call Works End-to-End

```
Claude Code (LLM)                    MCP Server                    Atlassian Cloud API
      |                                  |                                  |
      |-- tool call: getConfluencePage ->|                                  |
      |   {cloudId, pageId, format}      |                                  |
      |                                  |-- GET /wiki/api/v2/pages/{id} -->|
      |                                  |   with auth headers              |
      |                                  |                                  |
      |                                  |<-- JSON response (page body) ----|
      |                                  |                                  |
      |<-- tool result (JSON array) -----|                                  |
      |   [{type: "text", text: "..."}]  |                                  |
```

1. Claude Code decides to call a tool and emits a structured tool-call block with the tool name and parameters.
2. The Claude Code runtime routes the call to the Atlassian MCP server process.
3. The MCP server translates the call into the appropriate Atlassian REST API request, authenticating with stored OAuth/API tokens.
4. The response is converted into a JSON array of content blocks (typically `[{type: "text", text: "<page content>"}]`) and returned.
5. Claude Code receives the result as a tool output and can read/process it.

---

## The Context Cost Problem

### LLM Context Window Basics

Claude Code operates within a **context window** — the total amount of text (tokens) the model can process in a single conversation turn. Every message, tool call, and tool result occupies space in this window. When the window fills up, older content must be summarized or dropped.

### Why Atlassian Content Is Problematic

Confluence pages can be extremely large. The Tensix SFPU ISA page, for example, is approximately **175,000 characters** (~50,000+ tokens). For perspective:

- A typical tool result might be 500–5,000 characters
- A large code file might be 10,000–30,000 characters
- The SFPU ISA page is **175,000 characters** — roughly 10-35x a normal tool result

If this content were loaded directly into the context window, it would:
1. **Consume a massive portion of the available context**, leaving little room for the rest of the conversation
2. **Push out previous conversation history** via summarization
3. **Slow down processing** since the model must attend to all context on every turn
4. **Be wasteful** when only a small section of the page is actually needed

### The Overflow Mechanism

Claude Code has a built-in safeguard: when a tool result exceeds a maximum token threshold, the content is **not** injected into the context window. Instead:

1. The result is written to a **temporary file on disk** (e.g., `/home/user/.claude/projects/.../tool-results/mcp-atlassian-getConfluencePage-{timestamp}.txt`)
2. Claude Code receives a **short stub message** instead, containing:
   - The file path where the full content was saved
   - The total character count
   - The JSON schema of the saved data
   - Instructions to use `Read`, `Grep`, or `Bash` to extract specific portions

This means the 175K-character page costs only ~200–300 characters of context for the stub, instead of 175K.

### Practical Context Cost Breakdown

| Action | Context Cost |
|---|---|
| Calling `getConfluencePage` (small page, <30K chars) | Full page content loaded into context |
| Calling `getConfluencePage` (large page, >30K chars) | ~300 chars for the overflow stub |
| `Grep` on the overflow file | Only matching lines + context loaded |
| `Bash` with `python3` to extract a section | Only the printed output loaded |
| `Read` with `offset`/`limit` on the overflow file | Only the requested line range loaded |

The overflow mechanism transforms the problem from "load everything" to "load only what you need."

---

## The Usage Strategy and Why It Works This Way

### The Constraint (from CLAUDE.md)

The project configuration restricts Atlassian MCP usage to a single authoritative page:

- **Only use `getConfluencePage`** with page ID `1170505767` (Tensix SFPU ISA)
- **Do not use** `search`, `searchConfluenceUsingCql`, or navigate to other pages
- **Use `Grep` and `Bash`** to extract specific sections from the overflow file

### Why This Constraint Exists

1. **Scope control**: The SFPU ISA page is the sole authoritative source for instruction set details. Searching broadly could surface outdated or incorrect information from other pages.
2. **Predictability**: Always fetching the same page means the workflow is deterministic. The page ID is known, the cloud ID is known, and the content structure is stable.
3. **Cost efficiency**: Searching (`search` or `searchConfluenceUsingCql`) returns summary results that may require multiple follow-up `fetch` calls to find the right content, each consuming context and API round-trips. Going directly to the known page eliminates this exploration cost.

### The Three-Phase Workflow

#### Phase 1: Fetch (one-time, ~300 chars context cost)

```
Call: mcp__atlassian__getConfluencePage
  cloudId: "b9d94484-5dbd-4ae2-b670-6f414aefb4cd"
  pageId: "1170505767"
  contentFormat: "markdown"

Result: Overflow to /path/to/tool-results/mcp-atlassian-getConfluencePage-{timestamp}.txt
```

This is the only MCP call needed. The entire page is now cached on disk for the rest of the session.

#### Phase 2: Locate (surgical, ~200-500 chars per search)

Use `Grep` to find the position of relevant content:

```
Grep pattern="(?i)condition.?code" path="/path/to/overflow-file.txt"
```

Or use `Bash` with Python to find positions and surrounding context:

```python
import json, re
with open('/path/to/overflow-file.txt') as f:
    data = json.load(f)
text = data[0]['text']
matches = [(m.start(), m.end()) for m in re.finditer(r'condition.?code', text, re.I)]
```

This costs only the match results in context — typically a few hundred characters.

#### Phase 3: Extract (targeted, proportional to section size)

Use `Bash` with Python to extract only the needed section by character position:

```python
print(text[43500:50000])  # Extract ~6500 chars around the Condition Codes section
```

This loads only the relevant section into context. For a section like Condition Codes (~6,000 chars), this is **~3.5% of the full page**.

### Why Not Just Use `Read` Directly?

The overflow file is a JSON array, not plain text. The actual page content is inside `data[0]['text']` as a single string with escaped newlines (`\n`). Using `Read` with line offsets would work on the JSON structure (which is typically 1-4 very long lines), not on the logical lines of the markdown content. Python extraction via `Bash` gives precise control over character-level slicing within the content string.

### Cost Summary for a Typical Query

| Step | Tool | Context Added |
|---|---|---|
| Fetch page | `getConfluencePage` | ~300 chars (overflow stub) |
| Find section positions | `Bash` (python3) | ~500 chars (match positions) |
| Extract section content | `Bash` (python3) | ~6,000 chars (section text) |
| **Total** | | **~6,800 chars** |

Without the overflow strategy, the same query would inject **175,000 chars** into context. The targeted extraction approach uses **~4%** of what a naive approach would cost.

---

## Comparison: Naive vs. Targeted Approach

| Aspect | Naive (load full page) | Targeted (overflow + extract) |
|---|---|---|
| Context consumed | ~175,000 chars | ~6,800 chars |
| API calls | 1 | 1 (same) |
| Tool calls total | 1 | 3 (fetch + locate + extract) |
| Precision | Must scan everything | Jumps directly to relevant section |
| Reusability | Page lost if context rolls over | File persists on disk for entire session |
| Risk of context overflow | High | Low |

The disk-based overflow file also has an advantage over repeated `getConfluencePage` calls: the file persists for the entire session. If a second question about the same page comes in later, the fetch step is skipped entirely — only the locate and extract steps are needed, reducing cost further.
