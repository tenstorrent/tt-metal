(function () {
  "use strict";

  const CPP_KEYWORDS = new Set([
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "break", "case", "catch", "class", "compl", "concept", "const", "consteval",
    "constexpr", "constinit", "const_cast", "continue", "co_await", "co_return",
    "co_yield", "decltype", "default", "delete", "do", "dynamic_cast", "else",
    "enum", "explicit", "export", "extern", "false", "for", "friend", "goto",
    "if", "inline", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public",
    "register", "reinterpret_cast", "requires", "return", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "using", "virtual", "volatile", "while", "xor", "xor_eq"
  ]);

  const CPP_TYPES = new Set([
    "bool", "char", "char8_t", "char16_t", "char32_t", "double", "float", "int",
    "int8_t", "int16_t", "int32_t", "int64_t", "long", "short", "signed",
    "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "unsigned", "void",
    "wchar_t"
  ]);

  const PYTHON_KEYWORDS = new Set([
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "case", "class", "continue", "def", "del", "elif", "else",
    "except", "finally", "for", "from", "global", "if", "import", "in",
    "is", "lambda", "match", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield"
  ]);

  const PSEUDOCODE_KEYWORDS = new Set([
    "current", "expand", "optional_group", "repeat", "required", "times"
  ]);

  const PSEUDOCODE_TYPES = new Set([
    "MopConfig", "Template0", "Template1"
  ]);

  const PSEUDOCODE_INPUTS = new Set([
    "Count1", "Mask"
  ]);

  const PSEUDOCODE_FIELDS = new Set([
    "end_op", "end_op_shadow", "end_ops", "mid_ops", "op_a", "op_b", "op_c",
    "start_op", "start_op_shadow"
  ]);

  function appendToken(fragment, text, kind) {
    if (!text) return;
    if (!kind) {
      fragment.appendChild(document.createTextNode(text));
      return;
    }
    const span = document.createElement("span");
    span.className = `tok-${kind}`;
    span.textContent = text;
    fragment.appendChild(span);
  }

  function isIdentifierStart(char) {
    return /[A-Za-z_]/.test(char);
  }

  function isIdentifierPart(char) {
    return /[A-Za-z0-9_]/.test(char);
  }

  function highlightCpp(code, source) {
    const fragment = document.createDocumentFragment();
    let index = 0;
    let lineStart = true;

    while (index < source.length) {
      const char = source[index];
      const next = source[index + 1] || "";

      if (char === "\n") {
        appendToken(fragment, char);
        index += 1;
        lineStart = true;
        continue;
      }

      if (/\s/.test(char)) {
        const start = index;
        while (index < source.length && /\s/.test(source[index]) && source[index] !== "\n") index += 1;
        appendToken(fragment, source.slice(start, index));
        continue;
      }

      if (lineStart && char === "#") {
        const start = index;
        do {
          index = source.indexOf("\n", index);
          if (index === -1) {
            index = source.length;
            break;
          }
          index += 1;
        } while (source[index - 2] === "\\");
        appendToken(fragment, source.slice(start, index), "preprocessor");
        lineStart = true;
        continue;
      }

      lineStart = false;

      if (char === "/" && next === "/") {
        const end = source.indexOf("\n", index);
        const stop = end === -1 ? source.length : end;
        appendToken(fragment, source.slice(index, stop), "comment");
        index = stop;
        continue;
      }

      if (char === "/" && next === "*") {
        const end = source.indexOf("*/", index + 2);
        const stop = end === -1 ? source.length : end + 2;
        appendToken(fragment, source.slice(index, stop), "comment");
        index = stop;
        continue;
      }

      if (char === '"' || char === "'") {
        const quote = char;
        const start = index++;
        while (index < source.length) {
          if (source[index] === "\\") {
            index += 2;
          } else if (source[index++] === quote) {
            break;
          }
        }
        appendToken(fragment, source.slice(start, index), "string");
        continue;
      }

      if (/\d/.test(char)) {
        const start = index++;
        while (index < source.length && /[A-Za-z0-9_.]/.test(source[index])) index += 1;
        appendToken(fragment, source.slice(start, index), "number");
        continue;
      }

      if (isIdentifierStart(char)) {
        const start = index++;
        while (index < source.length && isIdentifierPart(source[index])) index += 1;
        const word = source.slice(start, index);
        let lookahead = index;
        while (lookahead < source.length && /\s/.test(source[lookahead])) lookahead += 1;
        let kind = null;
        if (CPP_KEYWORDS.has(word)) kind = "keyword";
        else if (CPP_TYPES.has(word)) kind = "type";
        else if (/^[A-Z][A-Z0-9_]+$/.test(word)) kind = "macro";
        else if (source.slice(index, index + 2) === "::") kind = "namespace";
        else if (source[lookahead] === "(") kind = "function";
        appendToken(fragment, word, kind);
        continue;
      }

      appendToken(fragment, char);
      index += 1;
    }

    code.replaceChildren(fragment);
  }

  function highlightPython(code, source) {
    const fragment = document.createDocumentFragment();
    let index = 0;

    while (index < source.length) {
      const char = source[index];

      if (/\s/.test(char)) {
        const start = index++;
        while (index < source.length && /\s/.test(source[index])) index += 1;
        appendToken(fragment, source.slice(start, index));
        continue;
      }

      if (char === "#") {
        const end = source.indexOf("\n", index);
        const stop = end === -1 ? source.length : end;
        appendToken(fragment, source.slice(index, stop), "comment");
        index = stop;
        continue;
      }

      if (char === '"' || char === "'") {
        const quote = char;
        const triple = source.slice(index, index + 3) === quote.repeat(3);
        const delimiter = triple ? quote.repeat(3) : quote;
        const start = index;
        index += delimiter.length;
        while (index < source.length) {
          if (source[index] === "\\") {
            index += 2;
          } else if (source.slice(index, index + delimiter.length) === delimiter) {
            index += delimiter.length;
            break;
          } else {
            index += 1;
          }
        }
        appendToken(fragment, source.slice(start, index), "string");
        continue;
      }

      if (/\d/.test(char)) {
        const start = index++;
        while (index < source.length && /[A-Za-z0-9_.]/.test(source[index])) index += 1;
        appendToken(fragment, source.slice(start, index), "number");
        continue;
      }

      if (isIdentifierStart(char)) {
        const start = index++;
        while (index < source.length && isIdentifierPart(source[index])) index += 1;
        const word = source.slice(start, index);
        let lookahead = index;
        while (lookahead < source.length && /\s/.test(source[lookahead])) lookahead += 1;
        const kind = PYTHON_KEYWORDS.has(word) ? "keyword" : source[lookahead] === "(" ? "function" : null;
        appendToken(fragment, word, kind);
        continue;
      }

      appendToken(fragment, char);
      index += 1;
    }

    code.replaceChildren(fragment);
  }

  function highlightPseudocode(code, source) {
    const fragment = document.createDocumentFragment();
    let index = 0;

    while (index < source.length) {
      const char = source[index];

      if (/\s/.test(char)) {
        const start = index++;
        while (index < source.length && /\s/.test(source[index])) index += 1;
        appendToken(fragment, source.slice(start, index));
        continue;
      }

      if (source.slice(index, index + 2) === "//") {
        const end = source.indexOf("\n", index);
        const stop = end === -1 ? source.length : end;
        appendToken(fragment, source.slice(index, stop), "comment");
        index = stop;
        continue;
      }

      if (/\d/.test(char)) {
        const start = index++;
        while (index < source.length && /[A-Za-z0-9_.]/.test(source[index])) index += 1;
        appendToken(fragment, source.slice(start, index), "number");
        continue;
      }

      if (isIdentifierStart(char)) {
        const start = index++;
        while (index < source.length && isIdentifierPart(source[index])) index += 1;
        const word = source.slice(start, index);
        let kind = null;
        if (PSEUDOCODE_KEYWORDS.has(word)) kind = "keyword";
        else if (PSEUDOCODE_TYPES.has(word)) kind = "type";
        else if (PSEUDOCODE_INPUTS.has(word)) kind = "input";
        else if (PSEUDOCODE_FIELDS.has(word)) kind = "field";
        appendToken(fragment, word, kind);
        continue;
      }

      const operator = [">>=", "=>", "=="].find(candidate => source.startsWith(candidate, index));
      if (operator) {
        appendToken(fragment, operator, "operator");
        index += operator.length;
        continue;
      }

      appendToken(fragment, char);
      index += 1;
    }

    code.replaceChildren(fragment);
  }

  function languageFor(code, source) {
    if (code.dataset.language) return code.dataset.language;
    if (/^\s*(python3|source|deactivate|ssh|curl)\b/m.test(source)) return "Shell";
    if (/^\s*(async\s+def|def)\s+[A-Za-z_]\w*\s*\(/m.test(source)) return "Python";
    return "C++";
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  function enhanceCodeBlocks() {
    for (const code of document.querySelectorAll("pre > code")) {
      if (code.dataset.enhanced === "true") continue;
      code.dataset.enhanced = "true";
      const source = code.textContent;
      const language = languageFor(code, source);
      code.dataset.editorLanguage = language;
      if (language === "C++") highlightCpp(code, source);
      else if (language === "Python") highlightPython(code, source);
      else if (language === "Pseudocode") highlightPseudocode(code, source);

      const pre = code.parentElement;
      const frame = document.createElement("div");
      frame.className = "code-frame";
      pre.parentNode.insertBefore(frame, pre);
      frame.appendChild(pre);

      const toolbar = document.createElement("div");
      toolbar.className = "code-toolbar";
      const label = document.createElement("span");
      label.textContent = language;
      const copy = document.createElement("button");
      copy.className = "code-copy";
      copy.type = "button";
      copy.textContent = "Copy";
      copy.addEventListener("click", async function () {
        try {
          await copyText(source);
          copy.textContent = "Copied";
        } catch (error) {
          copy.textContent = "Copy failed";
        }
        window.setTimeout(function () { copy.textContent = "Copy"; }, 1400);
      });
      toolbar.append(label, copy);
      frame.prepend(toolbar);
    }
  }

  function actionsForTopbar() {
    const topbar = document.querySelector(".topbar");
    if (!topbar) return null;
    let actions = topbar.querySelector(".docs-actions");
    if (actions) return actions;
    actions = document.createElement("div");
    actions.className = "docs-actions";
    topbar.appendChild(actions);
    return actions;
  }

  function activeTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function updateThemeButton(button) {
    const dark = activeTheme() === "dark";
    button.textContent = dark ? "Light" : "Dark";
    button.title = dark ? "Switch to light mode" : "Switch to dark mode";
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-pressed", String(dark));
  }

  function addThemeButton() {
    const actions = actionsForTopbar();
    if (!actions || actions.querySelector(".docs-theme")) return;
    const theme = document.createElement("button");
    theme.className = "docs-theme";
    theme.type = "button";
    updateThemeButton(theme);
    theme.addEventListener("click", function () {
      const next = activeTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { window.localStorage.setItem("hal-docs-theme", next); } catch (error) { /* Storage may be disabled. */ }
      updateThemeButton(theme);
    });
    actions.appendChild(theme);
  }

  function addRefreshButton() {
    const actions = actionsForTopbar();
    if (!actions || actions.querySelector(".docs-refresh")) return;
    const refresh = document.createElement("button");
    refresh.className = "docs-refresh";
    refresh.type = "button";
    refresh.textContent = "Refresh";
    refresh.title = "Regenerate the field catalog and reload this page";
    refresh.addEventListener("click", async function () {
      refresh.disabled = true;
      refresh.textContent = "Refreshing…";
      try {
        if (window.location.protocol === "http:" || window.location.protocol === "https:") {
          const response = await fetch("../../__refresh__", { method: "POST", cache: "no-store" });
          if (!response.ok) throw new Error(await response.text());
        }
        window.location.reload();
      } catch (error) {
        refresh.disabled = false;
        refresh.textContent = "Refresh failed";
        refresh.title = String(error);
        window.setTimeout(function () { refresh.textContent = "Refresh"; }, 1800);
      }
    });
    actions.appendChild(refresh);
  }

  // Single source of truth for the sidebar "Packages" navigation. Every page
  // and the catalog generator carry a static copy for no-JS viewing; this
  // list rebuilds them at load time so a new page needs only one entry here.
  const PACKAGE_NAV = [
    { href: "index.html", label: "Documentation home" },
    { href: "cfg_hal_interface.html", label: "hal::cfg" },
    { href: "cfg_field_catalog.html", label: "hal::cfg field catalog" },
    { href: "address_counters_hal_interface.html", label: "hal::address_counters" },
    { href: "math_counters_hal_interface.html", label: "hal::math_counters" },
    { href: "gpr_ops_hal_interface.html", label: "hal::gpr_ops" },
    { href: "move_hal_interface.html", label: "hal::move" },
    { href: "atomic_hal_interface.html", label: "hal::atomic" },
    { href: "mop_hal_interface.html", label: "hal::mop" },
    { href: "replay_hal_interface.html", label: "hal::replay" },
    { href: "sync_hal_interface.html", label: "hal::sync" },
    { href: "unpack_hal_interface.html", label: "hal::unpack" },
    { href: "fpu_hal_interface.html", label: "hal::fpu" },
    { href: "src_hal_interface.html", label: "hal::src" },
    { href: "dst_hal_interface.html", label: "hal::dst" },
    { href: "pack_hal_interface.html", label: "hal::pack" },
    { href: "nop_hal_interface.html", label: "hal::nop" },
    { href: "misc_hal_interface.html", label: "hal::misc" }
  ];

  function syncPackageNav() {
    const currentPage = window.location.pathname.split("/").pop() || "index.html";
    for (const heading of document.querySelectorAll(".sidebar nav h2")) {
      if (heading.textContent.trim() !== "Packages") continue;
      const stale = [];
      for (let node = heading.nextSibling; node && node.nodeName !== "H2"; node = node.nextSibling) {
        stale.push(node);
      }
      heading.after(...PACKAGE_NAV.map(entry => {
        const anchor = document.createElement("a");
        anchor.href = entry.href;
        anchor.textContent = entry.label;
        if (entry.href === currentPage) anchor.className = "current";
        return anchor;
      }));
      for (const node of stale) node.remove();
      break;
    }
  }

  const SIDEBAR_STATE_KEY = "hal-docs-sidebar-expanded";

  function storedSidebarExpanded() {
    try {
      const stored = window.localStorage.getItem(SIDEBAR_STATE_KEY);
      return stored === null ? true : stored !== "false";
    } catch (error) {
      return true;
    }
  }

  function updateSidebarState(layout, sidebar, button, expanded) {
    layout.classList.toggle("sidebar-expanded", expanded);
    layout.classList.toggle("sidebar-collapsed", !expanded);
    button.textContent = "☰";
    button.title = expanded ? "Collapse the sidebar" : "Expand the sidebar";
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-expanded", String(expanded));
    sidebar.dataset.expanded = String(expanded);
  }

  function enhanceSidebars() {
    const expanded = storedSidebarExpanded();
    for (const sidebar of document.querySelectorAll(".layout > .sidebar")) {
      const layout = sidebar.parentElement;
      if (!layout || sidebar.querySelector(".sidebar-toggle")) continue;

      layout.classList.add("sidebar-enhanced");
      const hasToc = Array.from(layout.children).some(child => child.classList.contains("toc"));
      layout.classList.toggle("sidebar-has-toc", hasToc);

      const button = document.createElement("button");
      button.className = "sidebar-toggle";
      button.type = "button";
      updateSidebarState(layout, sidebar, button, expanded);
      button.addEventListener("click", function () {
        const next = !layout.classList.contains("sidebar-expanded");
        updateSidebarState(layout, sidebar, button, next);
        try { window.localStorage.setItem(SIDEBAR_STATE_KEY, String(next)); } catch (error) { /* Storage may be disabled. */ }
      });
      sidebar.prepend(button);
    }
  }

  const DOCUMENT_ENDPOINT = "../../__document__";
  const DOCUMENT_DRAFT_PREFIX = "hal-docs-draft:";
  const DOCUMENT_LOCAL_PREFIX = "hal-docs-document:";
  const EDITABLE_BLOCKS = "h1, h2, h3, h4, h5, h6, p, pre > code";
  const documentState = {
    path: window.location.pathname,
    savedContent: null,
    updatedAt: null,
    loadError: null,
    persistence: "server"
  };

  function storageGet(key) {
    try { return window.localStorage.getItem(key); } catch (error) { return null; }
  }

  function storageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (error) { /* Storage may be disabled. */ }
  }

  function storageRemove(key) {
    try { window.localStorage.removeItem(key); } catch (error) { /* Storage may be disabled. */ }
  }

  function pageDocumentPath() {
    if (window.location.protocol !== "file:") return window.location.pathname;
    return `/docs/hal/${window.location.pathname.split("/").pop() || "index.html"}`;
  }

  function blockFingerprint(block) {
    const kind = /^H[1-6]$/.test(block.tagName) ? block.tagName.toLowerCase() : block.tagName === "CODE" ? "code" : "p";
    const source = `${kind}|${(block.textContent || "").replace(/\s+/g, " ").trim()}`;
    let hash = 5381;
    for (let index = 0; index < source.length; index += 1) {
      hash = ((hash << 5) + hash) ^ source.charCodeAt(index);
    }
    return (hash >>> 0).toString(36);
  }

  function markEditableBlocks(root) {
    // IDs derive from each block's kind and original content so saved edits
    // reattach only to unchanged blocks after the page is regenerated;
    // stale edits fall off instead of landing on the wrong block.
    const blocks = Array.from(root.querySelectorAll(EDITABLE_BLOCKS));
    const used = new Set(blocks.map(block => block.dataset.docBlock).filter(Boolean));
    for (const block of blocks) {
      if (block.dataset.docBlock) continue;
      const fingerprint = blockFingerprint(block);
      let id = `block-${fingerprint}`;
      for (let occurrence = 2; used.has(id); occurrence += 1) id = `block-${fingerprint}-${occurrence}`;
      block.dataset.docBlock = id;
      used.add(id);
    }
  }

  function escapeMarkdownText(value) {
    return value.replace(/([\\`*_\[\]])/g, "\\$1");
  }

  function inlineToMarkdown(element) {
    let markdown = "";
    for (const node of element.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        markdown += escapeMarkdownText(node.nodeValue || "");
        continue;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      const tag = node.tagName.toLowerCase();
      if (tag === "br") markdown += "  \n";
      else if (tag === "code") {
        const value = node.textContent || "";
        const fence = value.includes("`") ? "``" : "`";
        markdown += `${fence}${value}${fence}`;
      } else if (tag === "strong" || tag === "b") markdown += `**${inlineToMarkdown(node)}**`;
      else if (tag === "em" || tag === "i") markdown += `*${inlineToMarkdown(node)}*`;
      else if (tag === "a") markdown += `[${inlineToMarkdown(node)}](${node.getAttribute("href") || "#"})`;
      else markdown += inlineToMarkdown(node);
    }
    return markdown.trim();
  }

  function fenceFor(source) {
    const matches = source.match(/`+/g) || [];
    const longest = matches.reduce((size, match) => Math.max(size, match.length), 0);
    return "`".repeat(Math.max(3, longest + 1));
  }

  function markdownLanguage(code) {
    const language = code.dataset.editorLanguage || code.dataset.language || languageFor(code, code.textContent);
    return ({ "C++": "cpp", "Python": "python", "Shell": "shell", "Pseudocode": "pseudocode" })[language] || language.toLowerCase();
  }

  function blockToMarkdown(block) {
    if (/^H[1-6]$/.test(block.tagName)) {
      return `${"#".repeat(Number(block.tagName[1]))} ${inlineToMarkdown(block)}`;
    }
    if (block.tagName === "CODE") {
      const source = block.textContent.replace(/\n$/, "");
      const fence = fenceFor(source);
      return `${fence}${markdownLanguage(block)}\n${source}\n${fence}`;
    }
    return inlineToMarkdown(block);
  }

  function markdownFromMain(main) {
    markEditableBlocks(main);
    return Array.from(main.querySelectorAll("[data-doc-block]"), block =>
      `<!-- hal-doc:block:${block.dataset.docBlock} -->\n${blockToMarkdown(block)}`
    ).join("\n\n");
  }

  function pageBlockNearViewport(main) {
    const blocks = Array.from(main.querySelectorAll("[data-doc-block]"));
    if (!blocks.length) return null;
    const topbar = document.querySelector(".topbar");
    const focusLine = (topbar ? topbar.getBoundingClientRect().bottom : 0) + Math.min(160, window.innerHeight * .24);
    return blocks.reduce((closest, block) => {
      const rect = block.getBoundingClientRect();
      const distance = rect.top <= focusLine && rect.bottom >= focusLine ? 0 : Math.min(Math.abs(rect.top - focusLine), Math.abs(rect.bottom - focusLine));
      return !closest || distance < closest.distance ? { block, distance } : closest;
    }, null).block.dataset.docBlock;
  }

  function markdownRangeForBlock(markdown, id) {
    const marker = /<!--\s*hal-doc:block:([A-Za-z0-9_-]+)\s*-->/g;
    const matches = [];
    let match;
    while ((match = marker.exec(markdown)) !== null) {
      matches.push({ id: match[1], markerStart: match.index, contentStart: marker.lastIndex });
    }
    const index = matches.findIndex(candidate => candidate.id === id);
    if (index === -1) return null;
    let start = matches[index].contentStart;
    while (markdown[start] === "\r" || markdown[start] === "\n") start += 1;
    let end = index + 1 < matches.length ? matches[index + 1].markerStart : markdown.length;
    while (end > start && /\s/.test(markdown[end - 1])) end -= 1;
    return { start, end };
  }

  function blockAtMarkdownOffset(markdown, offset) {
    const marker = /<!--\s*hal-doc:block:([A-Za-z0-9_-]+)\s*-->/g;
    let match;
    let active = null;
    while ((match = marker.exec(markdown)) !== null && match.index <= offset) active = match[1];
    return active;
  }

  function blockLabel(root, id) {
    const block = Array.from(root.querySelectorAll("[data-doc-block]")).find(candidate => candidate.dataset.docBlock === id);
    if (!block) return "Current block";
    const kind = /^H[1-6]$/.test(block.tagName) ? block.tagName : block.tagName === "CODE" ? "Code" : "Paragraph";
    const summary = block.textContent.trim().replace(/\s+/g, " ");
    return `${kind} · ${summary.slice(0, 72)}${summary.length > 72 ? "…" : ""}`;
  }

  function parseEditorBlocks(markdown) {
    const marker = /<!--\s*hal-doc:block:([A-Za-z0-9_-]+)\s*-->/g;
    const blocks = [];
    let match;
    let previous = null;
    while ((match = marker.exec(markdown)) !== null) {
      if (!previous && markdown.slice(0, match.index).trim()) {
        throw new Error("Keep the first hal-doc block marker at the start of the document.");
      }
      if (previous) {
        blocks.push({ id: previous.id, content: markdown.slice(previous.start, match.index).trim() });
      }
      previous = { id: match[1], start: marker.lastIndex };
    }
    if (!previous) throw new Error("No editable block markers were found.");
    blocks.push({ id: previous.id, content: markdown.slice(previous.start).trim() });
    if (new Set(blocks.map(block => block.id)).size !== blocks.length) {
      throw new Error("Each hal-doc block marker must be unique.");
    }
    return blocks;
  }

  function safeLink(value) {
    const href = value.trim();
    if (/^(?:https?:|mailto:|#|\/|\.\.?\/)/i.test(href)) return href;
    return href.includes(":") ? "#" : href;
  }

  function appendInlineMarkdown(parent, source) {
    let index = 0;
    const appendText = value => parent.appendChild(parent.ownerDocument.createTextNode(value));
    while (index < source.length) {
      if (source[index] === "\\" && index + 1 < source.length) {
        appendText(source[index + 1]);
        index += 2;
        continue;
      }
      const link = source.slice(index).match(/^\[([^\]]+)\]\(([^)\s]+)\)/);
      if (link) {
        const anchor = parent.ownerDocument.createElement("a");
        anchor.href = safeLink(link[2]);
        appendInlineMarkdown(anchor, link[1]);
        parent.appendChild(anchor);
        index += link[0].length;
        continue;
      }
      const code = source.slice(index).match(/^(`+)([\s\S]*?)\1/);
      if (code) {
        const element = parent.ownerDocument.createElement("code");
        element.textContent = code[2];
        parent.appendChild(element);
        index += code[0].length;
        continue;
      }
      const strong = source.slice(index).match(/^\*\*([^\n]+?)\*\*/);
      if (strong) {
        const element = parent.ownerDocument.createElement("strong");
        appendInlineMarkdown(element, strong[1]);
        parent.appendChild(element);
        index += strong[0].length;
        continue;
      }
      const emphasis = source.slice(index).match(/^\*([^\n]+?)\*/);
      if (emphasis) {
        const element = parent.ownerDocument.createElement("em");
        appendInlineMarkdown(element, emphasis[1]);
        parent.appendChild(element);
        index += emphasis[0].length;
        continue;
      }
      if (source.startsWith("  \n", index)) {
        parent.appendChild(parent.ownerDocument.createElement("br"));
        index += 3;
        continue;
      }
      let stop = index + 1;
      while (stop < source.length && !/[\\`*\[]/.test(source[stop]) && !source.startsWith("  \n", stop)) stop += 1;
      appendText(source.slice(index, stop).replace(/\n/g, " "));
      index = stop;
    }
  }

  function codeLanguageLabel(value) {
    const normalized = value.trim().toLowerCase();
    if (["c++", "cpp", "cxx"].includes(normalized)) return "C++";
    if (["py", "python", "python3"].includes(normalized)) return "Python";
    if (["sh", "shell", "bash", "console"].includes(normalized)) return "Shell";
    if (normalized === "pseudocode") return "Pseudocode";
    return value.trim() || "Text";
  }

  function applyEditorMarkdown(root, markdown, strict) {
    markEditableBlocks(root);
    const targets = new Map(Array.from(root.querySelectorAll("[data-doc-block]"), block => [block.dataset.docBlock, block]));
    const blocks = parseEditorBlocks(markdown);
    if (strict) {
      const missing = Array.from(targets.keys()).filter(id => !blocks.some(block => block.id === id));
      const unknown = blocks.filter(block => !targets.has(block.id));
      if (missing.length || unknown.length) {
        throw new Error("Keep every hal-doc block marker unchanged so edits stay attached to the right content.");
      }
    }

    // Validate every block before mutating any, so a structural error cannot
    // leave the page with a half-applied mix of old and new content.
    const operations = [];
    for (const block of blocks) {
      const target = targets.get(block.id);
      if (!target) continue;
      if (/^H[1-6]$/.test(target.tagName)) {
        const heading = block.content.match(/^(#{1,6})[ \t]+([^\n]+)$/);
        if (!heading) throw new Error(`${block.id} must contain one Markdown heading.`);
        operations.push({ target, kind: "heading", heading });
      } else if (target.tagName === "CODE") {
        const fenced = block.content.match(/^(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)\n\1$/);
        if (!fenced) throw new Error(`${block.id} must contain one fenced code block.`);
        operations.push({ target, kind: "code", fenced });
      } else {
        operations.push({ target, kind: "inline", content: block.content });
      }
    }

    for (const operation of operations) {
      let target = operation.target;
      if (operation.kind === "heading") {
        const tagName = `h${operation.heading[1].length}`;
        if (target.tagName.toLowerCase() !== tagName) {
          const replacement = target.ownerDocument.createElement(tagName);
          for (const attribute of target.attributes) replacement.setAttribute(attribute.name, attribute.value);
          target.replaceWith(replacement);
          target = replacement;
        }
        target.replaceChildren();
        appendInlineMarkdown(target, operation.heading[2]);
      } else if (operation.kind === "code") {
        target.textContent = operation.fenced[3];
        target.dataset.language = codeLanguageLabel(operation.fenced[2]);
        target.dataset.editorLanguage = target.dataset.language;
      } else {
        target.replaceChildren();
        appendInlineMarkdown(target, operation.content);
      }
    }
    return blocks.length;
  }

  function unmatchedEditCount(root, markdown) {
    try {
      const ids = new Set(Array.from(root.querySelectorAll("[data-doc-block]"), block => block.dataset.docBlock));
      return parseEditorBlocks(markdown).filter(block => !ids.has(block.id)).length;
    } catch (error) {
      return 0;
    }
  }

  function noteEditIssue(main, message) {
    if (main.querySelector(".docs-editor-stale-notice")) return;
    const notice = document.createElement("div");
    notice.className = "docs-editor-stale-notice";
    notice.setAttribute("role", "status");
    notice.textContent = message;
    main.prepend(notice);
  }

  function readLocalDocument(key) {
    const raw = storageGet(key);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.content === "string") {
        return { content: parsed.content, savedAt: Date.parse(parsed.saved_at || "") || 0 };
      }
    } catch (error) { /* Copies from older builds are bare Markdown with no timestamp. */ }
    return { content: raw, savedAt: 0 };
  }

  function writeLocalDocument(key, content) {
    storageSet(key, JSON.stringify({ content, saved_at: new Date().toISOString() }));
  }

  async function loadDocumentEdit(main) {
    documentState.path = pageDocumentPath();
    const localKey = DOCUMENT_LOCAL_PREFIX + documentState.path;
    const local = readLocalDocument(localKey);
    if (!/^https?:$/.test(window.location.protocol)) {
      if (local) {
        documentState.savedContent = local.content;
        documentState.persistence = "browser";
      }
    } else {
      try {
        const endpoint = new URL(DOCUMENT_ENDPOINT, window.location.href);
        endpoint.searchParams.set("path", documentState.path);
        const response = await fetch(endpoint, { cache: "no-store" });
        if (!response.ok) {
          const error = new Error(`Editor state request failed (${response.status})`);
          error.status = response.status;
          throw error;
        }
        const payload = await response.json();
        if (payload.document) {
          documentState.savedContent = payload.document.content;
          documentState.updatedAt = payload.document.updated_at;
        }
      } catch (error) {
        documentState.loadError = error;
        documentState.persistence = "browser";
        console.warn("Could not load saved documentation edits", error);
      }
      if (local) {
        // The offline fallback copy wins only when the server copy is
        // unavailable, absent, or older; otherwise the shared server copy is
        // authoritative and the stale fallback is dropped so a later save
        // cannot overwrite someone else's newer edits with it.
        const serverUpdatedAt = Date.parse(documentState.updatedAt || "") || 0;
        if (documentState.loadError || !documentState.savedContent || local.savedAt > serverUpdatedAt) {
          documentState.savedContent = local.content;
          documentState.persistence = "browser";
        } else {
          storageRemove(localKey);
        }
      }
    }
    if (documentState.savedContent) {
      try {
        applyEditorMarkdown(main, documentState.savedContent, false);
        const unmatched = unmatchedEditCount(main, documentState.savedContent);
        if (unmatched) {
          noteEditIssue(main, `${unmatched} saved edit${unmatched === 1 ? "" : "s"} no longer match this page's structure and ${unmatched === 1 ? "was" : "were"} not applied. Saving from the editor refreshes the stored copy.`);
        }
      } catch (error) {
        documentState.loadError = error;
        noteEditIssue(main, `Saved edits could not be applied to this page: ${error.message} The page shows its original content.`);
        console.warn("Could not apply saved documentation edits", error);
      }
    }
  }

  async function persistDocumentEdit(content) {
    if (!/^https?:$/.test(window.location.protocol)) {
      writeLocalDocument(DOCUMENT_LOCAL_PREFIX + documentState.path, content);
      documentState.persistence = "browser";
      return "browser";
    }
    try {
      const response = await fetch(new URL(DOCUMENT_ENDPOINT, window.location.href), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: documentState.path, content })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (![404, 405, 501].includes(response.status)) {
          throw new Error(payload.message || `Save failed (${response.status})`);
        }
        writeLocalDocument(DOCUMENT_LOCAL_PREFIX + documentState.path, content);
        documentState.persistence = "browser";
        return "browser";
      }
      documentState.updatedAt = payload.document.updated_at;
      documentState.persistence = "server";
      storageRemove(DOCUMENT_LOCAL_PREFIX + documentState.path);
      return "server";
    } catch (error) {
      if (!(error instanceof TypeError)) throw error;
      writeLocalDocument(DOCUMENT_LOCAL_PREFIX + documentState.path, content);
      documentState.persistence = "browser";
      return "browser";
    }
  }

  function editorButton(label, className) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    return button;
  }

  function openDocumentEditor(main) {
    const draftKey = DOCUMENT_DRAFT_PREFIX + documentState.path;
    const generated = markdownFromMain(main);
    const recovered = storageGet(draftKey);
    const initial = recovered || generated;
    const previewBase = main.cloneNode(true);
    let activeBlockId = pageBlockNearViewport(main);
    let previewScrollFrame = null;
    const dialog = document.createElement("dialog");
    dialog.className = "docs-editor";
    dialog.setAttribute("aria-labelledby", "docs-editor-title");

    const header = document.createElement("header");
    header.className = "docs-editor-head";
    const headingGroup = document.createElement("div");
    const title = document.createElement("h2");
    title.id = "docs-editor-title";
    title.textContent = `Edit ${documentState.path.split("/").pop()}`;
    const hint = document.createElement("p");
    hint.textContent = "Edit headings, paragraphs, and fenced code. Keep the hal-doc block markers in place.";
    headingGroup.append(title, hint);
    const close = editorButton("Close", "docs-editor-close");
    header.append(headingGroup, close);

    const tabs = document.createElement("div");
    tabs.className = "docs-editor-tabs";
    tabs.setAttribute("role", "tablist");
    const writeTab = editorButton("Edit", "docs-editor-tab active");
    const previewTab = editorButton("Preview", "docs-editor-tab");
    writeTab.setAttribute("role", "tab");
    previewTab.setAttribute("role", "tab");
    writeTab.setAttribute("aria-selected", "true");
    previewTab.setAttribute("aria-selected", "false");
    const location = document.createElement("span");
    location.className = "docs-editor-location";
    tabs.append(writeTab, previewTab, location);

    const body = document.createElement("div");
    body.className = "docs-editor-body";
    const textarea = document.createElement("textarea");
    textarea.className = "docs-editor-textarea";
    textarea.value = initial;
    textarea.spellcheck = false;
    textarea.setAttribute("aria-label", "Page Markdown");
    const preview = document.createElement("div");
    preview.className = "docs-editor-preview";
    preview.hidden = true;
    body.append(textarea, preview);

    const footer = document.createElement("footer");
    footer.className = "docs-editor-footer";
    const status = document.createElement("span");
    status.className = "docs-editor-status";
    status.textContent = recovered ? "Recovered your unsaved draft." : documentState.persistence === "browser" ? "Static hosting detected; saved changes will persist in this browser." : "Drafts are saved in this browser as you type.";
    const footerActions = document.createElement("div");
    footerActions.className = "docs-editor-footer-actions";
    const discard = editorButton("Discard draft", "docs-editor-secondary");
    const save = editorButton("Save changes", "docs-editor-save");
    footerActions.append(discard, save);
    footer.append(status, footerActions);
    dialog.append(header, tabs, body, footer);
    document.body.appendChild(dialog);

    function updateLocation() {
      location.textContent = blockLabel(previewBase, activeBlockId);
      location.title = location.textContent;
    }

    function selectEditorBlock(id) {
      const range = markdownRangeForBlock(textarea.value, id);
      if (!range) return;
      activeBlockId = id;
      textarea.focus({ preventScroll: true });
      textarea.setSelectionRange(range.start, range.start);
      const lineCount = textarea.value.slice(0, range.start).split("\n").length - 1;
      const lineHeight = Number.parseFloat(window.getComputedStyle(textarea).lineHeight) || 20;
      textarea.scrollTop = Math.max(0, lineCount * lineHeight - textarea.clientHeight * .28);
      updateLocation();
    }

    function scrollPreviewToBlock() {
      const target = Array.from(preview.querySelectorAll("[data-doc-block]")).find(candidate => candidate.dataset.docBlock === activeBlockId);
      if (!target) return;
      for (const block of preview.querySelectorAll(".docs-editor-preview-active")) block.classList.remove("docs-editor-preview-active");
      target.classList.add("docs-editor-preview-active");
      const previewRect = preview.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      preview.scrollTop += targetRect.top - previewRect.top - preview.clientHeight * .22;
    }

    function previewBlockNearTop() {
      const blocks = Array.from(preview.querySelectorAll("[data-doc-block]"));
      if (!blocks.length) return null;
      const focusLine = preview.getBoundingClientRect().top + Math.min(140, preview.clientHeight * .22);
      return blocks.reduce((closest, block) => {
        const rect = block.getBoundingClientRect();
        const distance = rect.top <= focusLine && rect.bottom >= focusLine ? 0 : Math.min(Math.abs(rect.top - focusLine), Math.abs(rect.bottom - focusLine));
        return !closest || distance < closest.distance ? { block, distance } : closest;
      }, null).block.dataset.docBlock;
    }

    function setTab(showPreview) {
      writeTab.classList.toggle("active", !showPreview);
      previewTab.classList.toggle("active", showPreview);
      writeTab.setAttribute("aria-selected", String(!showPreview));
      previewTab.setAttribute("aria-selected", String(showPreview));
      textarea.hidden = showPreview;
      preview.hidden = !showPreview;
      if (!showPreview) {
        window.requestAnimationFrame(function () { selectEditorBlock(activeBlockId); });
        return;
      }
      preview.replaceChildren();
      try {
        const rendered = previewBase.cloneNode(true);
        applyEditorMarkdown(rendered, textarea.value, true);
        preview.appendChild(rendered);
        window.requestAnimationFrame(scrollPreviewToBlock);
        status.textContent = "Preview is up to date.";
      } catch (error) {
        const message = document.createElement("div");
        message.className = "docs-editor-error";
        message.textContent = error.message;
        preview.appendChild(message);
        status.textContent = "Fix the Markdown structure to preview or save.";
      }
    }

    textarea.addEventListener("input", function () {
      activeBlockId = blockAtMarkdownOffset(textarea.value, textarea.selectionStart) || activeBlockId;
      updateLocation();
      storageSet(draftKey, textarea.value);
      status.textContent = "Draft saved locally.";
    });
    for (const eventName of ["click", "keyup", "select"]) {
      textarea.addEventListener(eventName, function () {
        activeBlockId = blockAtMarkdownOffset(textarea.value, textarea.selectionStart) || activeBlockId;
        updateLocation();
      });
    }
    preview.addEventListener("click", function (event) {
      const block = event.target.closest("[data-doc-block]");
      if (!block || !preview.contains(block)) return;
      event.preventDefault();
      activeBlockId = block.dataset.docBlock;
      setTab(false);
    });
    preview.addEventListener("scroll", function () {
      if (previewScrollFrame !== null) window.cancelAnimationFrame(previewScrollFrame);
      previewScrollFrame = window.requestAnimationFrame(function () {
        previewScrollFrame = null;
        activeBlockId = previewBlockNearTop() || activeBlockId;
        updateLocation();
      });
    });
    writeTab.addEventListener("click", function () { setTab(false); });
    previewTab.addEventListener("click", function () { setTab(true); });
    close.addEventListener("click", function () { dialog.close(); });
    dialog.addEventListener("cancel", function (event) {
      event.preventDefault();
      dialog.close();
    });
    dialog.addEventListener("close", function () { dialog.remove(); });
    discard.addEventListener("click", function () {
      storageRemove(draftKey);
      textarea.value = generated;
      setTab(false);
      status.textContent = "Draft discarded.";
    });
    save.addEventListener("click", async function () {
      try {
        applyEditorMarkdown(previewBase.cloneNode(true), textarea.value, true);
        save.disabled = true;
        discard.disabled = true;
        save.textContent = "Saving…";
        status.textContent = "Writing this page to persistent storage…";
        const persistence = await persistDocumentEdit(textarea.value);
        storageRemove(draftKey);
        documentState.savedContent = textarea.value;
        status.textContent = persistence === "server" ? "Saved to the repository state file. Reloading…" : "Saved in this browser. Reloading…";
        window.location.reload();
      } catch (error) {
        save.disabled = false;
        discard.disabled = false;
        save.textContent = "Save changes";
        status.textContent = error.message;
      }
    });

    dialog.showModal();
    updateLocation();
    selectEditorBlock(activeBlockId);
  }

  function addEditButton(main) {
    const actions = actionsForTopbar();
    if (!actions || actions.querySelector(".docs-edit")) return;
    const edit = editorButton("Edit", "docs-edit");
    edit.title = documentState.updatedAt ? `Edit this page (last saved ${documentState.updatedAt})` : "Edit this page as Markdown";
    edit.addEventListener("click", function () { openDocumentEditor(main); });
    actions.prepend(edit);
  }

  async function initialize() {
    const main = document.querySelector("main");
    if (main) {
      markEditableBlocks(main);
      await loadDocumentEdit(main);
    }
    enhanceCodeBlocks();
    syncPackageNav();
    enhanceSidebars();
    addThemeButton();
    addRefreshButton();
    if (main) addEditButton(main);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
}());
