# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Build-time shaping of API reference pages.

This replaces what ``api_style.js`` used to do in every visitor's browser. The
work is structural — splitting a parameter entry into separately styleable
parts, turning an autosummary table into cards — so it belongs in the generated
HTML rather than in a client-side DOM pass:

* the styled result is in the static HTML, so it survives with JS disabled and
  is what crawlers and the search indexer see;
* there is no flash of unstyled API markup on load;
* nested inline markup is preserved. The JS version rebuilt rows from
  ``.textContent``, which flattened ``Defaults to :class:`None``` and dropped
  the ``<a>`` from cross-references inside a parameter description.

Nothing here hides content: the native ``dl.field-list`` keeps its structure and
its semantics, and only gains the wrapper elements that ``tt_theme.css`` needs
to give the description its own size and colour.
"""

from __future__ import annotations

from docutils import nodes
from sphinx import addnodes
from sphinx.ext.autosummary import autosummary_table

#: docutils renders the type/description separator as an en dash; a plain
#: hyphen shows up in docstrings that were written by hand.
_SEPARATORS = ("–", "—", "-")


def _is_separator(node: nodes.Node) -> bool:
    return isinstance(node, nodes.Text) and node.astext().strip() in _SEPARATORS


def _split_at_node(children: list[nodes.Node]):
    """Sphinx builds ``Parameters`` entries with the separator as its own node."""
    for index, child in enumerate(children):
        if _is_separator(child):
            return children[:index], children[index + 1 :]
    return None


def _split_inside_text(children: list[nodes.Node]):
    """``:returns:`` text is authored by hand, so the separator sits inside a
    Text node — ``<em>ttnn.Tensor</em>`` + ``" – the output tensor."``."""
    for index, child in enumerate(children):
        if not isinstance(child, nodes.Text):
            continue
        text = child.astext()
        for separator in _SEPARATORS:
            marker = f" {separator} "
            position = text.find(marker)
            if position == -1:
                continue
            before, after = text[:position], text[position + len(marker) :]
            head = children[:index]
            if before.strip():
                head = head + [nodes.Text(before)]
            tail = ([nodes.Text(after)] if after.strip() else []) + children[index + 1 :]
            return head, tail
    return None


def _split_entry(para: nodes.paragraph) -> bool:
    """Wrap ``name (type)`` and the description in their own inline elements.

    Returns True when the paragraph was restructured.
    """
    children = list(para.children)
    if not children:
        return False

    split = _split_at_node(children) or _split_inside_text(children)
    if split is None:
        return False

    head, tail = split
    if not head or not tail:
        return False

    para.clear()
    para += [
        nodes.inline("", "", *head, classes=["tt-api-param-name"]),
        nodes.inline("", "", *tail, classes=["tt-api-param-desc"]),
    ]
    return True


def _shape_field_lists(doctree: nodes.document) -> int:
    """Split every parameter entry of every API field list."""
    count = 0
    for field_list in doctree.findall(nodes.field_list):
        for field in field_list.findall(nodes.field):
            body = field.next_node(nodes.field_body)
            if body is None:
                continue

            items = list(body.findall(nodes.list_item))
            if items:
                # Parameters / Keyword Arguments / Raises: one entry per item.
                for item in items:
                    paragraph = item.next_node(nodes.paragraph)
                    if paragraph is not None and _split_entry(paragraph):
                        count += 1
            else:
                # Returns: a single paragraph, "type – description".
                paragraph = body.next_node(nodes.paragraph)
                if paragraph is not None and _split_entry(paragraph):
                    count += 1
    return count


def _shape_param_tables(doctree: nodes.document) -> int:
    """Turn Breathe's C++ parameter tables into the same row markup.

    Breathe emits parameters as a two- or three-column table. Reshaping it here
    keeps the C++ pages visually consistent with the Python ones without the
    browser having to read a table back into a list.
    """
    count = 0
    for desc in doctree.findall(addnodes.desc):
        if desc.get("domain") != "cpp":
            continue
        for table in list(desc.findall(nodes.table)):
            rows = list(table.findall(nodes.row))
            if not rows:
                continue

            container = nodes.container(classes=["tt-api-param-list"])
            for row in rows:
                cells = list(row.findall(nodes.entry))
                if len(cells) < 2:
                    continue
                name_cell, desc_cell = cells[0], cells[1]

                paragraph = nodes.paragraph(classes=["tt-api-param-row"])
                paragraph += nodes.inline("", "", *_detached(name_cell), classes=["tt-api-param-name"])
                paragraph += nodes.inline("", "", *_detached(desc_cell), classes=["tt-api-param-desc"])
                container += paragraph
                count += 1

            if len(container.children):
                table.replace_self(container)
    return count


def _detached(cell: nodes.Element) -> list[nodes.Node]:
    """Return a cell's inline content, unwrapped from its paragraph."""
    out: list[nodes.Node] = []
    for child in cell.children:
        if isinstance(child, nodes.paragraph):
            out.extend(child.children)
        else:
            out.append(child)
    return out


def _shape_autosummary(doctree: nodes.document) -> int:
    """Turn autosummary tables into a list of cards.

    The signature arguments trailing the name are dropped here: they are a bare
    text node in the rendered HTML, so CSS alone could not remove them.
    """
    count = 0
    # Match the wrapper autosummary emits rather than the table's classes: a
    # table inside a docstring Note carries no marker of its own, and must not
    # be touched.
    for wrapper in list(doctree.findall(autosummary_table)):
        cards = nodes.container(classes=["tt-api-card-list"])
        for row in wrapper.findall(nodes.row):
            cells = list(row.findall(nodes.entry))
            if not cells:
                continue

            card = nodes.container(classes=["tt-api-card"])

            name = nodes.paragraph(classes=["tt-api-card-name"])
            reference = cells[0].next_node(nodes.reference)
            if reference is not None:
                name += reference.deepcopy()
            else:
                name += nodes.Text(cells[0].astext().split("(")[0].strip())
            card += name

            if len(cells) > 1 and cells[1].astext().strip():
                card += nodes.paragraph("", "", *_detached(cells[1]), classes=["tt-api-card-desc"])

            cards += card
            count += 1

        if len(cards.children):
            wrapper.replace_self(cards)
    return count


def on_doctree_resolved(app, doctree: nodes.document, docname: str) -> None:
    _shape_field_lists(doctree)
    _shape_param_tables(doctree)
    _shape_autosummary(doctree)


def setup(app):
    app.connect("doctree-resolved", on_doctree_resolved)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
