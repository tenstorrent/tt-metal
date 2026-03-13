# How MCP Servers Work (Glean)

## What is MCP?

MCP (Model Context Protocol) is a standardized protocol that allows AI assistants like Claude to interact with external services through a defined interface. Think of it as a **plugin system** — it extends capabilities by giving access to tools hosted by external servers.

## Architecture

The architecture has three layers:

```
┌─────────────┐       ┌──────────────┐       ┌─────────────────┐
│ Claude (LLM) │ ───── │  MCP Client  │ ───── │  MCP Server     │
│             │       │  (in Claude  │       │  (Glean)        │
│             │       │   Code CLI)  │       │                 │
└─────────────┘       └──────────────┘       └─────────────────┘
```

1. **MCP Server** — An external process (local or remote) that exposes a set of **tools** and **resources** via the MCP protocol. Glean is one such server.
2. **MCP Client** — Built into Claude Code (the CLI). It connects to configured MCP servers at startup, discovers their tools, and makes them available to the LLM.
3. **The LLM (Claude)** — Sees the MCP tools listed alongside all other tools (Bash, Read, Edit, etc.) and can call them like any other tool.

## How Glean Specifically Works

Glean is an MCP server that provides AI-powered enterprise knowledge search. It indexes **all company knowledge sources** — SharePoint, Slack, Confluence, Google Drive, Jira, email, and more. When a session starts, Claude Code connects to the Glean MCP server (after authentication), and Claude receives a description of its tools.

### Authentication

Glean requires the user to authenticate before any tools become available. When the MCP connection is established, the user sees a message like:

```
Authentication successful. Connected to glean_default.
```

The Glean server uses the user's identity to filter all results based on their access permissions. Claude only sees documents the user is authorized to view. If a user asks about a document they cannot access, the results will be empty — this is expected behavior, not an error.

### Supported Data Sources

Glean indexes content from many enterprise applications. The `app` filter parameter accepts these values:

| App ID | Platform |
|---|---|
| `o365sharepoint` | Microsoft SharePoint |
| `slack` | Slack |
| `confluence` | Confluence |
| `gdrive` | Google Drive |
| `jira` | Jira |
| `github` | GitHub |
| `gmailnative` | Gmail |
| `notion` | Notion |
| `o365` | Microsoft 365 (general) |
| `greenhouse` | Greenhouse (recruiting) |
| `airtable` | Airtable |
| `tableau` | Tableau |
| `trello` | Trello |
| `datadog` | Datadog |

And others including `azure`, `smartsheet`, `lucid`, `testrail`, `jfrog`, `clubhouse`, `evernote`, `googlesites`, `invision`, `zeplin`, `bynder`.

---

## Tools Exposed by Glean

Glean exposes three tools, each serving a different purpose. They form a natural workflow: **search** discovers documents, **read_document** fetches their full content, and **chat** provides AI-synthesized answers.

### Naming Convention

All tool names follow the pattern `mcp__glean_default__<tool_name>`. The `mcp__` prefix identifies it as an MCP tool, `glean_default` identifies the Glean server instance, and the rest is the tool name.

---

### 1. `search` — Keyword-Based Document Discovery

Full name: `mcp__glean_default__search`

This tool performs a keyword search across all indexed company knowledge sources. It returns a list of matching documents with metadata, text snippets, and URLs. It does **not** return full document content — only enough to identify, rank, and preview results.

#### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Short, targeted keywords. Use `*` to match all documents (useful with filters). |
| `app` | string | No | Filter to a specific app (see Supported Data Sources above) |
| `type` | string | No | Filter by document type: `pull`, `spreadsheet`, `slides`, `email`, `direct message`, `folder` |
| `from` | string | No | Documents updated/commented by a person. Accepts a name, `"me"`, or `"myteam"`. |
| `owner` | string | No | Documents created by a person. Accepts a name, `"me"`, or `"myteam"`. |
| `updated` | string | No | Relative time filter: `"today"`, `"yesterday"`, `"past_week"`, `"past_2_weeks"`, `"past_month"`, or a month name like `"March"` |
| `after` | string | No | Documents updated after this date (exclusive). Format: `"YYYY-MM-DD"`. |
| `before` | string | No | Documents updated before this date (exclusive). Format: `"YYYY-MM-DD"`. |
| `channel` | string | No | Filter to a specific Slack channel. Only use when the user explicitly wants content from a channel. |
| `sort_by_recency` | boolean | No | Sort by newest first instead of relevance. Only use when the user's intent is "latest" or "most recent". |
| `exhaustive` | boolean | No | Return all matching results. Only use when the user requests "all", "each", or "every" result. |
| `dynamic_search_result_filters` | string | No | Narrow results using filters discovered from a previous search. Format: `key:value\|key2:value2`. Never invent filters — only use keys/values seen in prior results' `matching_filters`. |

#### Query Guidelines

The search engine is **keyword-based**, not natural language. Follow these rules:

- **Use 2-5 targeted keywords**: `Tensix ISA SFPU` — not `What is the Tensix ISA and how does the SFPU work?`
- **No boolean logic**: Do not use `OR` or `AND` operators.
- **No query stuffing**: Do not add synonyms, categories, or descriptive verbs. If the user asks "what is X", query `X`.
- **No quotes**: Unless the user is searching for an exact phrase or verbatim string.
- **No duplicate terms**: Each keyword should appear once.
- **Use `*` with filters**: To find all documents matching a filter (e.g., all spreadsheets by a person), use `*` as the query with appropriate filters.

#### Examples

**Finding documents about a topic on SharePoint:**

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "Tensix ISA instruction set",
    "app": "o365sharepoint"
  }
}
```

**Finding recent Slack messages from a specific person:**

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "kernel performance regression",
    "app": "slack",
    "from": "John Smith",
    "updated": "past_week"
  }
}
```

**Finding all Jira tickets owned by my team updated this month:**

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "*",
    "app": "jira",
    "owner": "myteam",
    "updated": "past_month",
    "exhaustive": true
  }
}
```

**Finding spreadsheets about a topic:**

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "performance benchmarks Wormhole",
    "type": "spreadsheet"
  }
}
```

**Finding the most recent documents about a topic:**

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "Ascalon architecture",
    "sort_by_recency": true
  }
}
```

**Narrowing results with dynamic filters from a prior search:**

If a first search returned results with `matching_filters` containing `site: ["SOC"]` and `folder: ["Tensix ISA"]`, you can narrow a follow-up search:

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "SFPU instructions",
    "dynamic_search_result_filters": "site:SOC|folder:Tensix ISA"
  }
}
```

#### Response Structure

Each result in the response contains:

```
document
├── title                  — Document title
├── url                    — Direct link to the document
├── snippets[]             — Matching text excerpts (most useful for quick answers)
├── datasource             — Source app (e.g. "o365sharepoint")
├── path                   — Folder path within the app
├── createTime             — When the document was created (ISO 8601)
├── updateTime             — Last update time (ISO 8601)
├── owner                  — Creator (name + obfuscatedId)
├── updatedBy              — Last editor (name + obfuscatedId)
├── matchingFilters        — Filter facets: app, site, folder, department, type
├── percentRetrieved       — How much of the document was indexed (e.g. "6.99%")
├── similarResults         — Nested related documents (increases payload significantly)
│   ├── numResults
│   └── visibleResults[]   — Each with its own full metadata envelope
└── (metadata envelope)    — datasourceId, datasourceInstance, documentId,
                             documentCategory, superContainerId, parentDocument, etc.
```

The **snippets** and **title/url** are the most immediately useful fields. The metadata envelope makes up the bulk of the payload but is rarely needed for answering user questions.

#### Context Cost

A typical search returns 10-15 results. Each carries a metadata envelope of ~2-4KB. When results include `similarResults`, nested documents multiply the payload. Total response size: **20-40KB**, of which useful content (snippets + titles + URLs) is ~2-4KB.

---

### 2. `read_document` — Fetch Full Document Content

Full name: `mcp__glean_default__read_document`

This tool retrieves the complete content of one or more documents by URL. It converts the source document (Word, PDF, HTML, etc.) into readable text. Use it after `search` identifies documents that need deeper reading beyond what the snippets provide.

#### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `urls` | string[] | Yes | List of document URLs to fetch. Batch multiple URLs in a single call. |

#### Behaviors

- **All-or-nothing**: Each URL returns either the full document or an empty response. There is no partial retrieval.
- **Batch multiple URLs**: If you need to read 3 documents, pass all 3 URLs in one call. Do not make 3 separate calls.
- **Idempotent**: Re-reading the same URL returns identical results. No need to retry.
- **Empty response**: Does not mean the document doesn't exist. Possible causes:
  - The user lacks access permissions.
  - The document type is not supported for content extraction.
  - The document is in a format Glean cannot parse.
- **URL variations**: Do not re-read the same URL with minor variations (e.g., different query parameters). Results will be identical or empty.

#### Examples

**Reading a single SharePoint document:**

```json
{
  "name": "mcp__glean_default__read_document",
  "parameters": {
    "urls": [
      "https://tenstorrent.sharepoint.com/sites/SOC/_layouts/15/Doc.aspx?sourcedoc=%7BF657471C-C01A-470E-BB33-26F53C2CF1D6%7D&file=Tensix_ISA_draft.docx"
    ]
  }
}
```

**Batch reading multiple documents found via search:**

```json
{
  "name": "mcp__glean_default__read_document",
  "parameters": {
    "urls": [
      "https://tenstorrent.sharepoint.com/sites/SOC/_layouts/15/Doc.aspx?sourcedoc=...&file=Tensix_ISA_draft.docx",
      "https://tenstorrent.sharepoint.com/sites/Software/_layouts/15/Doc.aspx?sourcedoc=...&file=CKernel_Guide.docx",
      "https://tenstorrent.sharepoint.com/sites/Tensix/_layouts/15/Doc.aspx?sourcedoc=...&file=SFPU_Overview.pptx"
    ]
  }
}
```

**Reading an external/public URL:**

`read_document` can also fetch external web pages, not just internal company documents:

```json
{
  "name": "mcp__glean_default__read_document",
  "parameters": {
    "urls": [
      "https://riscv.org/technical/specifications/"
    ]
  }
}
```

#### Typical Workflow

1. Run `search` to find relevant documents.
2. Examine the snippets. If they answer the question, stop.
3. If a document needs full reading, take its `url` from the search result and pass it to `read_document`.

#### Context Cost

Depends entirely on the document size. A short document might be 5KB; a comprehensive architecture spec could be 50KB+. Unlike `search` and `chat`, the response is primarily useful content with minimal metadata overhead.

---

### 3. `chat` — AI-Powered Question Answering

Full name: `mcp__glean_default__chat`

This is the most powerful tool. It sends a natural-language question to Glean's AI assistant, which internally searches across all indexed company knowledge, reads relevant documents, and synthesizes a grounded answer. Think of it as having Glean's AI do the `search` → `read_document` → `synthesize` workflow on your behalf.

#### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | The question or prompt. Unlike `search`, this accepts full natural-language sentences. |
| `context` | string[] | No | Previous messages for multi-turn conversation. Each string is one previous message, included in order before the current message. |

#### When to Use `chat` vs `search`

| Use `chat` when... | Use `search` when... |
|---|---|
| The question requires **analysis or synthesis** across multiple sources | You need **specific documents** or **exact results with metadata** |
| You need **contextual understanding**, not just document retrieval | The answer is likely in a **single document's snippet** |
| You want **reasoning and connections** between pieces of information | You need to **filter by app, person, date, or type** |
| You're asking **follow-up questions** building on previous context | You want to **control what sources are searched** |

#### Examples

**Asking about an internal architecture:**

```json
{
  "name": "mcp__glean_default__chat",
  "parameters": {
    "message": "What is the Tensix NEO Instruction Set Architecture and what are its key components?"
  }
}
```

**Asking about a process:**

```json
{
  "name": "mcp__glean_default__chat",
  "parameters": {
    "message": "What is Tenstorrent's export compliance process for new chip designs?"
  }
}
```

**Multi-turn follow-up with context:**

```json
{
  "name": "mcp__glean_default__chat",
  "parameters": {
    "message": "How does the SFPU handle predicated execution specifically?",
    "context": [
      "Tell me about the Tensix NEO ISA",
      "The Tensix NEO ISA is a custom instruction set for AI accelerator cores with TRISC cores and an SFPU SIMD engine."
    ]
  }
}
```

**Asking about people and organizational knowledge:**

```json
{
  "name": "mcp__glean_default__chat",
  "parameters": {
    "message": "Who is working on the Quasar SFPU and what are the recent design decisions?"
  }
}
```

#### Response Structure

The response is a JSON object:

```
{
  chatId: "..."                     — Session ID
  chatSessionTrackingToken: "..."   — Tracking token
  followUpPrompts: [...]            — Suggested follow-up questions (useful)
  messages: [
    {
      author: "GLEAN_AI"
      messageType: "UPDATE"         — Intermediate status messages (e.g. "Searching...")
      fragments: [
        { text: "Searching company knowledge" }
      ]
    },
    {
      author: "GLEAN_AI"
      messageType: "UPDATE"         — Shows what search queries Glean ran internally
      fragments: [
        { text: "**Searching:** " },
        { querySuggestion: { query: "..." } }
      ]
    },
    {
      author: "GLEAN_AI"
      messageType: "UPDATE"         — Shows which documents Glean is reading
      fragments: [
        { text: "**Reading:** " },
        { structuredResults: [ { document: { ... full metadata ... } } ] },
        { structuredResults: [ { document: { ... full metadata ... } } ] },
        ...                         — One structuredResults block per source document
      ]
    },
    {
      author: "GLEAN_AI"            — THE ACTUAL ANSWER (final message)
      fragments: [
        { text: "Based on the documents... [the synthesized answer]" }
      ]
    }
  ]
}
```

The **final message's `text` fragment** contains the actual answer. Everything else — the intermediate "Searching..." and "Reading..." update messages with their `structuredResults` — is metadata overhead. Each `structuredResults` block contains a full document metadata envelope (same as in `search` results).

#### Extracting the Answer

Because the response is often too large for Claude's tool result token limit, a common pattern is to extract just the answer text:

```bash
python3 -c "
import json
with open('path/to/result.txt') as f:
    data = json.load(f)
for m in data.get('messages', []):
    for frag in m.get('fragments', []):
        t = frag.get('text', '')
        if len(t) > 200:  # Skip short status messages
            print(t)
"
```

To also extract the follow-up prompts:

```bash
python3 -c "
import json
with open('path/to/result.txt') as f:
    data = json.load(f)
print('Answer:')
for m in data.get('messages', []):
    for frag in m.get('fragments', []):
        t = frag.get('text', '')
        if len(t) > 200:
            print(t)
print()
print('Follow-up suggestions:')
for p in data.get('followUpPrompts', []):
    print(f'  - {p}')
"
```

#### Context Cost

This is the most expensive tool by far:

| Component | Typical Size |
|---|---|
| AI-generated answer | 1-3KB |
| Intermediate status messages | 1-2KB |
| Source document metadata (structuredResults) | 50-85KB |
| **Total response** | **50-90KB** |
| **Overhead ratio** | **~95-98%** |

The response frequently exceeds Claude's 25K token tool result limit, triggering a file-save fallback that requires additional Bash-based extraction.

---

## Tool Comparison at a Glance

| Tool | Input Style | Response Size | Useful Content | Best For |
|---|---|---|---|---|
| `search` | Keywords + filters | 20-40KB | Snippets, titles, URLs | Document discovery, quick answers from snippets |
| `read_document` | URLs | Varies | Full document text | Deep reading of specific documents |
| `chat` | Natural language | 50-90KB | Synthesized answer | Cross-document analysis, complex questions |

## How Claude Uses These Tools (Concrete Flow)

Here is what happens step-by-step when Claude uses Glean:

### 1. Authentication (at session start)

Glean requires the user to authenticate. When the MCP connection is established, the user sees `Authentication successful. Connected to glean_default.` From this point on, all queries are filtered by the user's access permissions.

### 2. Tool Discovery (at session start)

After authentication, Claude Code asks the Glean server: "What tools do you offer?" The server responds with JSON schemas describing each tool's name, description, and parameters. These get injected into Claude's system prompt alongside built-in tools.

### 3. Tool Invocation (during conversation)

When Claude decides to call a tool, it emits a structured function call:

```json
{
  "name": "mcp__glean_default__search",
  "parameters": {
    "query": "Tensix ISA",
    "app": "o365sharepoint"
  }
}
```

### 4. Request Routing (MCP Client)

Claude Code's MCP client intercepts this call, strips the `mcp__glean_default__` prefix to identify the server and tool, then forwards the request to the Glean server using the MCP protocol (typically JSON-RPC over stdio or HTTP/SSE).

### 5. Server Processing (Glean side)

The Glean server receives the request and processes it:
- **`search`**: Runs a keyword search against its index, applies permission filters, ranks results, and returns document metadata with snippets.
- **`read_document`**: Fetches the document by URL, converts it to readable text (from Word, PDF, HTML, etc.), and returns the content.
- **`chat`**: Runs a full RAG pipeline — internally searches for relevant documents, reads them, and generates a synthesized answer using Glean's AI model.

### 6. Response Return

The server sends back a JSON response, which Claude Code passes to Claude as the tool result. If the response exceeds Claude's context limits, it gets saved to a temporary file on disk, and Claude must read and parse the file to extract the answer.

## Practical Considerations

### Recommended Usage Patterns

**Most efficient** — Use `search` with targeted keywords and read the snippets directly:
```json
{ "query": "Tensix ISA SFPU instructions", "app": "o365sharepoint" }
```
The snippets embedded in search results are often sufficient to answer questions without further calls.

**Moderate** — Use `search` to find URLs, then `read_document` for specific documents:
```json
// Step 1: search to find the document
{ "query": "Tensix ISA draft" }
// Step 2: read the full document
{ "urls": ["https://tenstorrent.sharepoint.com/..."] }
```

**Most expensive** — Use `chat` for synthesis across multiple documents:
```json
{ "message": "Compare the SFPU and FPU instruction sets in the Tensix architecture" }
```
Reserve this for questions that genuinely require cross-document analysis.

### Handling Large Responses

When `chat` responses exceed Claude's tool result token limits (25K tokens), the response is saved to a temporary file. The file path is provided in the error/redirect message. Claude must then use Bash with Python JSON parsing to extract the answer (see the extraction scripts in the `chat` section above).

### Query Optimization for `search`

- **Short keywords**: 2-5 targeted terms.
- **Use filters**: Narrow by `app`, `from`, `owner`, `updated` to reduce noise.
- **Use `dynamic_search_result_filters`**: After a first search, use `matchingFilters` from the results to narrow follow-up searches.
- **Avoid `exhaustive` unless needed**: It significantly increases response size.
- **Avoid `sort_by_recency` by default**: Only use when the user explicitly wants the latest/newest content.

## How MCP Tools Differ From Built-in Tools

| Aspect | Built-in Tools (Read, Bash, etc.) | Glean MCP Tools |
|---|---|---|
| **Execution** | Handled directly by Claude Code | Delegated to the Glean server |
| **Availability** | Always present | Only if configured and authenticated |
| **Scope** | Local filesystem, shell | All indexed company knowledge |
| **Trust** | First-party | Depends on server configuration |
| **Permission model** | OS-level file permissions | Enterprise identity-based access control |
| **Confidentiality** | User controls what files are read | All returned information is internal company data |

## Summary

Glean is an enterprise knowledge MCP server that gives Claude access to the full breadth of company information across all tools and platforms. Its three tools form a workflow from lightweight discovery (`search`) through full content retrieval (`read_document`) to AI-powered synthesis (`chat`). The most efficient pattern is to use `search` with targeted keywords and rely on the returned snippets, falling back to `read_document` for full content or `chat` for cross-document synthesis only when needed.

For a comparison of Glean with the DeepWiki MCP server, see [mcp_servers_comparison.md](mcp_servers_comparison.md).
