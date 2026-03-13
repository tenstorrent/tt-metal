# MCP Server Comparison: DeepWiki vs Glean

This document compares the two MCP servers available in this project — **DeepWiki** and **Glean** — covering their scope, response characteristics, context efficiency, and best-use scenarios.

## Overview

| Aspect | DeepWiki | Glean |
|---|---|---|
| **Purpose** | AI-powered documentation for GitHub repositories | Enterprise knowledge search across all company tools |
| **Data source** | Single GitHub repository (or up to 10 for cross-repo questions) | SharePoint, Slack, Confluence, Google Drive, Jira, email, and more |
| **Authentication** | None required | Enterprise SSO required |
| **Permission model** | Public repos (or configured private repos) | Filtered by the authenticated user's enterprise access permissions |

## Tool Mapping

Both servers expose three tools, but they map to different workflows:

| Purpose | DeepWiki Tool | Glean Tool |
|---|---|---|
| Browse available content | `read_wiki_structure` | `search` (with broad query or `*`) |
| Get full content | `read_wiki_contents` | `read_document` |
| Ask a targeted question | `ask_question` | `chat` |

The tools are not exact equivalents. DeepWiki's `read_wiki_structure` returns a lightweight topic map, while Glean's closest equivalent is a broad `search` query that returns document listings with metadata. DeepWiki's `ask_question` returns a clean text answer, while Glean's `chat` returns an answer embedded in a large JSON structure with full source metadata.

## Response Format

This is the most significant practical difference.

### DeepWiki

DeepWiki responses are **plain text** optimized for human and LLM consumption. An `ask_question` call returns essentially a markdown-formatted answer with minimal wrapper:

```
Question → DeepWiki RAG pipeline → Clean text answer
```

The response payload is almost entirely useful content.

### Glean

Glean responses are **structured JSON** designed for enterprise search UIs. Every response carries a full metadata envelope per document:

```json
{
  "documents": [
    {
      "title": "Tensix_ISA_draft",
      "url": "https://tenstorrent.sharepoint.com/...",
      "snippets": ["The SFPU is a SIMD engine..."],
      "datasource": "o365sharepoint",
      "datasourceId": "Drives_b!ZKXrrR6B5kGFJ89wK0ZE...",
      "datasourceInstance": "o365sharepoint",
      "documentCategory": "COLLABORATIVE_CONTENT",
      "documentId": "O365SHAREPOINT_Drives_b!ZKXrrR6B5kGFJ89wK0ZE...",
      "matchingFilters": { "app": [...], "site": [...], "folder": [...], "department": [...] },
      "owner": { "name": "...", "obfuscatedId": "..." },
      "updatedBy": { "name": "...", "obfuscatedId": "..." },
      "parentDocument": { "datasource": "...", "id": "...", "title": "...", "url": "..." },
      "superContainerId": "O365SHAREPOINT_Site_tenstorrent.sharepoint.com,...",
      "percentRetrieved": "6.99%",
      "similarResults": { "numResults": 4, "visibleResults": [ ... ] }
    }
  ]
}
```

The useful content (title, snippet, URL) is roughly 5-10% of the payload. The rest is enterprise metadata.

## Context Cost Comparison

| Operation | DeepWiki | Glean |
|---|---|---|
| Topic/document discovery | ~2-5KB (`read_wiki_structure`) | ~20-40KB (`search`) |
| Full content retrieval | ~10-50KB (`read_wiki_contents`) | Varies (`read_document`) |
| Targeted question | ~2-5KB (`ask_question`) | ~50-90KB (`chat`) |
| **Overhead ratio** | **~5-10% metadata** | **~90-98% metadata** |

A concrete example from this project: asking about the Tensix ISA through Glean's `chat` tool returned **86KB of JSON**, of which the actual AI answer was **~1.5KB**. The equivalent DeepWiki `ask_question` call would have returned ~2-5KB total with the answer as the primary content.

### Why Glean Responses Overflow

Glean's `chat` response frequently exceeds Claude's tool result token limit (25K tokens). When this happens:

1. The response gets saved to a temporary file on disk.
2. Claude must use Bash with Python JSON parsing to extract the actual answer text.
3. This adds an extra round-trip and additional context consumption for the extraction code.

DeepWiki responses rarely hit this limit because they return clean text without metadata bloat.

## Strengths and Weaknesses

### DeepWiki

| Strengths | Weaknesses |
|---|---|
| Clean, context-efficient responses | Limited to GitHub repository content |
| No authentication needed | Cannot access SharePoint, Slack, Confluence, etc. |
| Direct answer extraction | No access to internal company documents |
| Low overhead ratio (~5-10%) | Cannot search across multiple knowledge sources |
| Cross-repo queries (up to 10 repos) | No permission-based filtering |

### Glean

| Strengths | Weaknesses |
|---|---|
| Searches across all company knowledge sources | Heavy metadata in every response (~90-98% overhead) |
| Enterprise permission filtering | Requires authentication |
| Covers SharePoint, Slack, Confluence, Jira, etc. | `chat` responses frequently overflow context limits |
| Rich filtering (by app, person, date, type) | Keyword-based search (not natural language) |
| Full document retrieval via URL | Answer extraction requires JSON parsing workarounds |

## When to Use Which

### Use DeepWiki when:
- The question is about **code in a GitHub repository** (architecture, APIs, implementation patterns)
- You need a **context-efficient** answer that won't consume a large portion of the context window
- You want to **compare concepts across repositories** (up to 10)
- The information exists in the codebase or its documentation

### Use Glean when:
- The information lives in **company documents** (SharePoint, Confluence, Google Drive)
- You need to find **internal presentations, specs, or compliance documents**
- You need to search **Slack messages or email threads**
- The question involves **people** (who wrote what, who owns what)
- The information is **not in any GitHub repository**

### Use both when:
- A question spans code-level details (DeepWiki) and internal documentation (Glean)
- You need to cross-reference a GitHub implementation with an internal design spec

## Efficiency Tips

1. **Start with DeepWiki** if the question might be answerable from the repository. It is 10-20x cheaper on context.
2. **Use Glean `search` before `chat`**. Search snippets are often sufficient and cost 2-4x less context than `chat`.
3. **Avoid Glean `chat` unless necessary**. Reserve it for questions that genuinely require cross-document synthesis.
4. **Use Glean filters aggressively** (`app`, `from`, `updated`) to narrow results and reduce payload size.
5. **Never call both `search` and `chat` for the same question** unless `search` snippets were genuinely insufficient.
