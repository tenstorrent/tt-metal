/* api_style.js — transforms API pages to match Figma design (node 238:7535 + 55-4153) */
(function () {
  "use strict";

  /* ─── helpers ─────────────────────────────────────────────────── */
  function copyBtn() {
    var btn = document.createElement("button");
    btn.className = "tt-api-copy-btn";
    btn.title = "Copy";
    var iconCopy =
      '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">' +
      '<rect x="7" y="7" width="9" height="9" rx="1.5" stroke="#678583" stroke-width="1.5"/>' +
      '<path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-7A1.5 1.5 0 0 0 3 5.5v7A1.5 1.5 0 0 0 4.5 14H6" stroke="#678583" stroke-width="1.5" stroke-linecap="round"/>' +
      "</svg>";
    var iconDone =
      '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">' +
      '<path d="M4 10l4 4 8-8" stroke="#1e86a9" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>";
    btn.innerHTML = iconCopy;
    btn.addEventListener("click", function () {
      var box = btn.closest(".tt-api-sig-box");
      var text = box ? box.textContent.replace(/\s+/g, " ").trim() : "";
      navigator.clipboard && navigator.clipboard.writeText(text);
      btn.innerHTML = iconDone;
      setTimeout(function () { btn.innerHTML = iconCopy; }, 1500);
    });
    return btn;
  }

  function makeSection(title) {
    var wrap = document.createElement("div");
    wrap.className = "tt-api-section";
    var heading = document.createElement("p");
    heading.className = "tt-api-section-heading";
    heading.textContent = title;
    wrap.appendChild(heading);
    return wrap;
  }

  function makeParamRow(nameHtml, typeText, descText) {
    var row = document.createElement("div");
    row.className = "tt-api-param-row";
    var nameP = document.createElement("p");
    nameP.className = "tt-api-param-name";
    var nameSpan = document.createElement("span");
    nameSpan.className = "tt-param-name";
    nameSpan.textContent = nameHtml;
    nameP.appendChild(nameSpan);
    if (typeText) {
      var typeSpan = document.createElement("span");
      typeSpan.className = "tt-param-type";
      typeSpan.textContent = " (" + typeText + ")";
      nameP.appendChild(typeSpan);
    }
    row.appendChild(nameP);
    if (descText) {
      var descP = document.createElement("p");
      descP.className = "tt-api-param-desc";
      descP.textContent = descText;
      row.appendChild(descP);
    }
    return row;
  }

  /* ─── Python dl.py.data / dl.py.function (TTNN autodoc) ──────── */
  function transformPyData() {
    var blocks = document.querySelectorAll(
      "dl.py.data, dl.py.function, dl.py.class, dl.py.method"
    );
    blocks.forEach(function (dl) {
      var dt = dl.querySelector("dt.sig");
      var dd = dl.querySelector("dd");
      if (!dt || !dd) return;

      /* 1 – Signature box */
      dt.classList.add("tt-api-sig-box");
      var hl = dt.querySelector("a.headerlink");
      if (hl) hl.replaceWith(copyBtn());

      /* 2 – Description: second <p> or first non-signature-like <p> */
      var allP = Array.from(dd.querySelectorAll(":scope > p"));
      /* Skip first paragraph if it looks like raw signature continuation */
      var descP = allP.find(function (p, i) {
        var txt = p.textContent.trim();
        if (p.classList.contains("rubric")) return false;
        /* First para often contains raw signature args — skip if no spaces and has colons */
        if (i === 0 && txt.indexOf(":") !== -1 && txt.indexOf(" ") === -1) return false;
        return txt.length > 0 && txt.length < 500;
      });
      if (descP) descP.classList.add("tt-api-description");

      /* 3 – field-list (Parameters / Keyword Arguments / Returns / Raises / …)
       *
       * A Sphinx field list is a flat sequence of <dt>label</dt><dd>value</dd>
       * pairs. The field-odd / field-even classes are only row parity — NOT the
       * field's meaning — so we must read each field's real label from its <dt>
       * and transform every pair, otherwise fields get mislabeled and later
       * ones (e.g. Returns, Raises) are silently dropped. */
      Array.from(dd.querySelectorAll("dl.field-list")).forEach(function (fieldList) {
        Array.from(fieldList.querySelectorAll(":scope > dt")).forEach(function (dtEl) {
          /* Pair each <dt> with its following <dd>. */
          var ddEl = dtEl.nextElementSibling;
          while (ddEl && ddEl.tagName.toLowerCase() !== "dd") {
            ddEl = ddEl.nextElementSibling;
          }
          if (!ddEl) return;

          var label = dtEl.textContent.replace(/[:\s]+$/, "").trim();
          var section = makeSection(label);
          var list = document.createElement("div");
          list.className = "tt-api-param-list";

          var items = ddEl.querySelectorAll("li");
          if (items.length) {
            /* List-style field (Parameters, Keyword Arguments): one row per item. */
            items.forEach(function (li) {
              var p = li.querySelector("p") || li;
              var strong = p.querySelector("strong");
              var ems = p.querySelectorAll("em");
              var name = strong ? strong.textContent.trim() : "";
              var typeText = Array.from(ems)
                .map(function (e) { return e.textContent.trim(); })
                .filter(function (t) { return t && t !== ","; })
                .join(", ")
                .replace(/,\s*,/g, ",");
              var fullText = p.textContent;
              var dash = fullText.indexOf("–"); /* en-dash – */
              var desc = dash !== -1 ? fullText.slice(dash + 1).trim() : "";
              /* Also try hyphen-minus fallback */
              if (!desc) {
                dash = fullText.indexOf(" - ");
                desc = dash !== -1 ? fullText.slice(dash + 3).trim() : "";
              }
              list.appendChild(makeParamRow(name, typeText, desc));
            });
          } else {
            /* Scalar field (Returns, Return type, Raises): a single value. */
            var full = ddEl.textContent.trim();
            var d = full.indexOf("–");
            var nm = d !== -1 ? full.slice(0, d).trim() : full;
            var ds = d !== -1 ? full.slice(d + 1).trim() : "";
            list.appendChild(makeParamRow(nm, "", ds));
          }

          section.appendChild(list);
          fieldList.parentNode.insertBefore(section, fieldList);
        });

        /* Replace the raw field-list with the transformed sections. */
        fieldList.parentNode.removeChild(fieldList);
      });
    });
  }

  /* ─── C++ dl.cpp.* (tt-metalium breathe) ─────────────────────── */
  function transformCpp() {
    var blocks = document.querySelectorAll(
      "dl.cpp.function, dl.cpp.type, dl.cpp.struct, dl.cpp.enum"
    );
    blocks.forEach(function (dl) {
      var dt = dl.querySelector("dt.sig");
      var dd = dl.querySelector("dd");
      if (!dt || !dd) return;

      dt.classList.add("tt-api-sig-box");
      var hl = dt.querySelector("a.headerlink");
      if (hl) hl.replaceWith(copyBtn());

      var paragraphs = Array.from(dd.querySelectorAll(":scope > p"));
      var descPara = null, returnPara = null;
      paragraphs.forEach(function (p) {
        var text = p.textContent.trim();
        if (text.startsWith("Return value:")) { returnPara = p; }
        else if (!descPara && text.length > 0) { descPara = p; }
      });
      if (descPara) descPara.classList.add("tt-api-description");

      var table = dd.querySelector("table.docutils");
      if (table) {
        var rows = table.querySelectorAll("tbody tr");
        if (rows.length) {
          var section = makeSection("Parameters");
          var list = document.createElement("div");
          list.className = "tt-api-param-list";
          rows.forEach(function (row) {
            var cells = row.querySelectorAll("td");
            var name = (cells[0] && cells[0].textContent.trim()) || "";
            var desc = (cells[1] && cells[1].textContent.trim()) || "";
            var type = (cells[2] && cells[2].textContent.trim()) || "";
            list.appendChild(makeParamRow(name, type, desc));
          });
          section.appendChild(list);
          var tableContainer = table.closest("p") || table;
          tableContainer.parentNode.insertBefore(section, tableContainer);
          tableContainer.parentNode.removeChild(tableContainer);
        }
      }

      if (returnPara) {
        var retText = returnPara.textContent.replace(/Return value:\s*/i, "").trim();
        var retSection = makeSection("Returns");
        var retList = document.createElement("div");
        retList.className = "tt-api-param-list";
        retList.appendChild(makeParamRow(retText, "", ""));
        retSection.appendChild(retList);
        returnPara.parentNode.insertBefore(retSection, returnPara);
        returnPara.parentNode.removeChild(returnPara);
      }
    });
  }

  /* ─── Example blocks → Figma 240:8481 ───────────────────────── */
  function transformExamples() {
    var iconCopy =
      '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">' +
      '<rect x="7" y="7" width="9" height="9" rx="1.5" stroke="#678583" stroke-width="1.5"/>' +
      '<path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-7A1.5 1.5 0 0 0 3 5.5v7A1.5 1.5 0 0 0 4.5 14H6" stroke="#678583" stroke-width="1.5" stroke-linecap="round"/>' +
      "</svg>";
    var iconDone =
      '<svg width="20" height="20" viewBox="0 0 20 20" fill="none">' +
      '<path d="M4 10l4 4 8-8" stroke="#1e86a9" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>";

    document.querySelectorAll("p.rubric").forEach(function (rubric) {
      var codeEl = rubric.nextElementSibling;
      if (!codeEl) return;

      /* Wrap rubric + code block in .tt-example-block */
      var wrapper = document.createElement("div");
      wrapper.className = "tt-example-block";

      /* Header bar */
      var header = document.createElement("div");
      header.className = "tt-example-header";
      header.textContent = rubric.textContent.trim();

      rubric.parentNode.insertBefore(wrapper, rubric);
      wrapper.appendChild(header);
      wrapper.appendChild(codeEl);   /* moves codeEl into wrapper */
      rubric.parentNode.removeChild(rubric);

      /* Add line numbers + copy button inside .highlight */
      var highlight = wrapper.querySelector(".highlight");
      if (highlight) {
        var pre = highlight.querySelector("pre");
        if (pre) {
          /* Split innerHTML by newlines — Pygments never spans tokens across lines */
          var html = pre.innerHTML.replace(/\n$/, ""); /* trim trailing newline */
          var htmlLines = html.split("\n");

          /* Build table: lineno | code per row */
          var table = document.createElement("table");
          table.className = "tt-example-table";

          htmlLines.forEach(function (lineHtml, idx) {
            var tr = document.createElement("tr");

            var tdNum = document.createElement("td");
            tdNum.className = "tt-lineno";
            tdNum.textContent = idx + 1;

            var tdCode = document.createElement("td");
            tdCode.className = "tt-codeline";
            tdCode.innerHTML = lineHtml || " "; /* nbsp for empty lines */

            tr.appendChild(tdNum);
            tr.appendChild(tdCode);
            table.appendChild(tr);
          });

          highlight.innerHTML = "";
          highlight.style.padding = "0";
          highlight.style.border = "none";
          highlight.appendChild(table);
        }

        /* Copy button — absolute top-right */
        var btn = document.createElement("button");
        btn.className = "tt-example-copy-btn";
        btn.title = "Copy code";
        btn.innerHTML = iconCopy;
        btn.addEventListener("click", function () {
          var tds = highlight.querySelectorAll("td.tt-codeline");
          var clean = Array.from(tds).map(function (td) {
            /* Strip either doctest prompt (>>> or ... continuation) so copied
             * multi-line examples paste as runnable Python. */
            return td.textContent.replace(/^(>>>|\.\.\.)\s?/, "");
          }).join("\n").trim();
          navigator.clipboard && navigator.clipboard.writeText(clean);
          btn.innerHTML = iconDone;
          setTimeout(function () { btn.innerHTML = iconCopy; }, 1500);
        });
        highlight.style.position = "relative";
        highlight.appendChild(btn);
      }
    });
  }

  /* ─── API listing page: autosummary tables → Figma cards ─────── */
  function transformApiIndex() {
    var tables = document.querySelectorAll("table.autosummary");
    tables.forEach(function (table) {
      var container = document.createElement("div");
      container.className = "tt-api-card-list";

      table.querySelectorAll("tbody tr").forEach(function (row) {
        var cells = row.querySelectorAll("td");
        if (!cells.length) return;

        var card = document.createElement("div");
        card.className = "tt-api-card";

        /* Function name — preserve link */
        var nameCell = cells[0];
        var link = nameCell.querySelector("a");
        var nameDiv = document.createElement("div");
        nameDiv.className = "tt-api-card-name";
        if (link) {
          var a = document.createElement("a");
          a.href = link.href;
          a.textContent = link.textContent.trim();
          nameDiv.appendChild(a);
        } else {
          nameDiv.textContent = nameCell.textContent.trim();
        }
        card.appendChild(nameDiv);

        /* Description */
        if (cells[1] && cells[1].textContent.trim()) {
          var descDiv = document.createElement("div");
          descDiv.className = "tt-api-card-desc";
          descDiv.textContent = cells[1].textContent.trim();
          card.appendChild(descDiv);
        }

        container.appendChild(card);
      });

      table.parentNode.insertBefore(container, table);
      table.parentNode.removeChild(table);
    });
  }

  /* ─── Breadcrumbs — Figma node 239:8072 ─────────────────────── */
  function transformBreadcrumbs() {
    /* Replace home icon link with plain "Home" text */
    var homeLink = document.querySelector(".wy-breadcrumbs a.icon-home");
    if (homeLink) {
      homeLink.textContent = "Home";
      homeLink.className = "";
    }

    /* Remove the "View page source" aside */
    var aside = document.querySelector(".wy-breadcrumbs-aside");
    if (aside) aside.parentNode && aside.parentNode.removeChild(aside);

    /* Remove the hr */
    var nav = document.querySelector('div[role="navigation"][aria-label="Page navigation"]');
    if (nav) {
      var hr = nav.querySelector("hr");
      if (hr) hr.parentNode && hr.parentNode.removeChild(hr);
    }
  }

  function run() {
    transformBreadcrumbs();
    transformPyData();
    transformCpp();
    transformExamples();
    transformApiIndex();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
