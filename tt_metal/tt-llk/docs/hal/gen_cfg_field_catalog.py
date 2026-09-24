#!/usr/bin/env python3
"""Generate the Blackhole hal::cfg field catalog from descriptor headers.

Field descriptions are copied verbatim from the trailing comments on
``static constexpr Field`` declarations.  Keep public_entry_point() in sync
with any hand-written descriptor facade added to the generated headers.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "tt_llk_blackhole/llk_lib/hal/cfg"
OUTPUT = Path(__file__).resolve().parent / "cfg_field_catalog.html"
FAMILIES = ("thread", "alu", "pack", "unpack", "thcon", "global")


@dataclass(frozen=True)
class FieldRecord:
    family: str
    source: Path
    line: int
    owners: tuple[str, ...]
    name: str
    args: tuple[str, ...]
    comment: str | None

    @property
    def declaration(self) -> str:
        return "::".join((*self.owners, self.name))


def compact(value: str) -> str:
    return " ".join(value.split())


def split_initializer(initializer: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}

    for index, char in enumerate(initializer):
        if char in depths:
            depths[char] += 1
        elif char in closing and depths[closing[char]]:
            depths[closing[char]] -= 1
        elif char == "," and not any(depths.values()):
            parts.append(compact(initializer[start:index]))
            start = index + 1
    parts.append(compact(initializer[start:]))
    if parts[-1] == "":
        parts.pop()
    return tuple(parts)


def code_without_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def parse_header(family: str) -> list[FieldRecord]:
    source = CFG_DIR / f"{family}.h"
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[FieldRecord] = []
    stack: list[tuple[str, int]] = []
    pending_struct: str | None = None
    depth = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        structural_text = line
        code = code_without_line_comment(line)

        # clang-format places a struct's opening brace on the following line.
        # Remember that declaration so fields retain their owning descriptor.
        if pending_struct is not None:
            if "{" in code:
                opening_depth = depth + code[: code.index("{") + 1].count("{")
                stack.append((pending_struct, opening_depth))
            pending_struct = None

        struct_match = re.match(r"\s*struct\s+(\w+)", code)
        if struct_match:
            if "{" in code:
                opening_depth = depth + code[: code.index("{") + 1].count("{")
                stack.append((struct_match.group(1), opening_depth))
            elif ";" not in code:
                pending_struct = struct_match.group(1)

        if "static constexpr Field" in code:
            start_line = index + 1
            statement = line
            while ";" not in code_without_line_comment(statement.splitlines()[-1]):
                index += 1
                if index >= len(lines):
                    raise RuntimeError(
                        f"unterminated Field declaration at {source}:{start_line}"
                    )
                statement += "\n" + lines[index]
            structural_text = statement

            match = re.search(
                r"static\s+constexpr\s+Field\s+(\w+)\s*\{(.*?)\}\s*;\s*(?:// ?(.*))?\s*$",
                statement,
                flags=re.DOTALL,
            )
            if not match:
                raise RuntimeError(
                    f"cannot parse Field declaration at {source}:{start_line}"
                )
            args = split_initializer(match.group(2))
            if len(args) != 8:
                raise RuntimeError(
                    f"expected eight Field initializer arguments at {source}:{start_line}; got {len(args)}"
                )
            comment = match.group(3)
            # Trailing descriptions wrap onto following comment-only lines;
            # append them so multi-line descriptor comments stay complete.
            if comment is not None:
                while index + 1 < len(lines):
                    continuation = re.fullmatch(r"\s*// ?(.*?)\s*", lines[index + 1])
                    if not continuation:
                        break
                    comment = f"{comment.rstrip()} {continuation.group(1)}"
                    index += 1
            records.append(
                FieldRecord(
                    family=family,
                    source=source,
                    line=start_line,
                    owners=tuple(owner for owner, _ in stack),
                    name=match.group(1),
                    args=args,
                    comment=comment if comment is not None else None,
                )
            )

        structural_code = "\n".join(
            code_without_line_comment(part) for part in structural_text.splitlines()
        )
        depth += structural_code.count("{") - structural_code.count("}")
        while stack and depth < stack[-1][1]:
            stack.pop()
        index += 1

    expected = sum("static constexpr Field" in line for line in lines)
    if len(records) != expected:
        raise RuntimeError(f"parsed {len(records)} of {expected} fields from {source}")
    return records


def public_entry_point(field: FieldRecord) -> tuple[str, str]:
    """Return the usable public expression and any index-range note."""
    owner = "::".join(field.owners)
    name = field.name

    if field.family == "thcon":
        if owner == "ThconTileDescriptorFields":
            return f"Thcon[Reg0].TileDescriptor.{name}", "section: Sec::S0..S1"
        match = re.fullmatch(r"ThconReg(\d+)Fields", owner)
        if match:
            return f"Thcon[Reg{match.group(1)}].{name}", "section: Sec::S0..S1"

    if field.family == "pack":
        address_ctrl = {
            "Pck0AddrCtrlXyReg0": "Packer[0].AddrCtrl[PackerReg::Reg0]",
            "Pck0AddrCtrlZwReg0": "Packer[0].AddrCtrl[PackerReg::Reg0]",
            "Pck0AddrCtrlXyReg1": "Packer[0].AddrCtrl[PackerReg::Reg1]",
            "Pck0AddrCtrlZwReg1": "Packer[0].AddrCtrl[PackerReg::Reg1]",
        }
        if owner in address_ctrl:
            register_set = "0" if owner.endswith("Reg0") else "1"
            return (
                f"{address_ctrl[owner]}.{name}",
                f"packer 0; register set {register_set}",
            )
        address_base = {
            "Pck0AddrBaseReg0": "PackerReg::Reg0",
            "Pck0AddrBaseReg1": "PackerReg::Reg1",
        }
        if owner in address_base:
            register_set = "0" if owner.endswith("Reg0") else "1"
            return (
                f"Packer[0].AddrBase[{address_base[owner]}]",
                f"packer 0; register set {register_set}",
            )
        if owner == "TileRowSetMappingRow::Fields":
            return "TileRowSetMapping[M][S]", "M: 0..3; S: 0..15"
        if owner == "TileFaceSetMappingRow::Fields":
            return "TileFaceSetMapping[M][S]", "M: 0..3; S: 0..15"

    if field.family == "unpack":
        if owner == "UnpackerAddrCtrlFields::Fields":
            return f"Unpacker[U].AddrCtrl[R].{name}", "U: 0..1; R: Reg0 or Reg1"
        if owner == "UnpackerAddrBaseFields::Fields":
            return "Unpacker[U].AddrBase[R]", "U: 0..1; R: Reg0 or Reg1"
        if owner == "UnpackerBlobsYStartFields":
            return f"Unpacker[0].BlobsYStart[BlobContext::{name}]", "unpacker 0 only"
        if owner == "UnpackerContextDescriptor":
            return "Unpacker[U].Cntx[C].Base", "U=0: C 0..7; U=1: C 0..1"
        if owner == "UnpackerFields::Fields":
            return f"Unpacker[U].{name}", "U: 0..1"

    if field.family == "thread":
        if owner == "DisableImpliedFmtFields":
            selector = {"SrcAField": "SrcA", "SrcBField": "SrcB"}[name]
            return f"DisableImpliedFmt[{selector}]", ""
        if owner == "AddrModFields::SrcFields":
            return (
                f"AddrMod[SrcA].{name} / AddrMod[SrcB].{name}",
                "section: Sec::S0..S7",
            )
        if owner == "AddrModFields":
            prefixes = {
                "Dest": "AddrMod[Dest]",
                "Fidelity": "AddrMod[Fidelity]",
                "Bias": "AddrMod[Bias]",
                "YSrc": "AddrMod[Src][Y]",
                "YDst": "AddrMod[Dest][Y]",
                "ZSrc": "AddrMod[Src][Z]",
                "ZDst": "AddrMod[Dest][Z]",
            }
            for prefix, entry in prefixes.items():
                if name.startswith(prefix):
                    section_count = (
                        "Sec::S0..S7"
                        if prefix in {"Dest", "Fidelity", "Bias"}
                        else "Sec::S0..S3"
                    )
                    return f"{entry}.{name[len(prefix):]}", f"section: {section_count}"
        if owner == "PerfCntCmdEntry::Fields":
            return f"PerfCntCmd[I].{name}", "I: 0..3"

    return f"{owner}::{name}", section_note(field)


def section_note(field: FieldRecord) -> str:
    count, stride = field.args[6], field.args[7]
    if count == "1":
        return ""
    if count == "2" and stride == "1536":
        return "section: Sec::S0..S1"
    return f"sections: {count}; stride: {stride} bits"


def section_usages(field: FieldRecord, group: str) -> tuple[str, ...]:
    """Describe what every physical section of a repeated descriptor controls."""
    count = int(field.args[6])
    if count == 1:
        return ()

    # Thread address modifiers are tables.  An instruction's addr_mode selects
    # one of the eight general slots; PACR's AddrMode selects one of the four
    # packer Y/Z slots.
    if group.startswith("AddrMod["):
        if group.endswith("[Y]") or group.endswith("[Z]"):
            return tuple(
                f"Packer address-modifier slot {section}; selected by PACR AddrMode = {section}."
                for section in range(count)
            )
        return tuple(
            f"Address-modifier slot {section}; selected when an instruction uses addr_mode = {section}."
            for section in range(count)
        )

    # These four-section thread registers are indexed banks, not copies tied to
    # a specific RISC.  Callers choose the entry whose programmed stream they
    # want an execution unit or TRISC-side access to use.
    if group == "StreamIdSync":
        return tuple(
            f"Sync-EXU stream-selector entry {section}." for section in range(count)
        )
    if group == "StreamIdTrisc":
        return tuple(
            f"TRISC stream-mapping entry {section}." for section in range(count)
        )

    # RiscDestAccessCtrl is the exceptional packed ALU register whose YAML
    # register description explicitly assigns sections 0, 1, and 2 to TRISCs.
    if group == "RiscDestAccessCtrl":
        return (
            "TRISC0 (Unpack) destination-register access control.",
            "TRISC1 (Math) destination-register access control.",
            "TRISC2 (Pack) destination-register access control.",
        )

    # Each of these four state-CFG words configures the correspondingly numbered
    # packer or packer register set.
    if group == "PackCounters":
        return tuple(
            f"Packer {section} counter and tile-position configuration."
            for section in range(count)
        )
    if group == "PackConcatMask":
        return tuple(
            f"Packer {section} concatenation mask." for section in range(count)
        )
    if group == "DestTargetRegCfgPack":
        return tuple(
            f"Packer destination register set {section}." for section in range(count)
        )

    # THCON is laid out as two 48-word hardware instances.  Packer fields within
    # an instance refer to the register-set pair named by the particular THCON
    # register (for example Reg1 uses 0/2 while Reg8 uses 1/3), so keep that
    # register-specific distinction visible in the per-field source description.
    if group.startswith("Thcon["):
        return (
            "First THCON hardware instance: Unpacker 0 / Mover 0 side; for Packer fields, the lower-numbered register set named by this THCON register.",
            "Second THCON hardware instance: Unpacker 1 / Mover 1 side; for Packer fields, the higher-numbered register set named by this THCON register.",
        )

    # The global arrays are physically consecutive state-CFG words.  Their
    # section number is the array index, except for the explicitly per-TRISC
    # register arrays.
    if group == "IntDescaleValues":
        return tuple(
            f"Integer descale-value entry {section}; contains four packed 8-bit values."
            for section in range(count)
        )
    if group == "TriscEndPc":
        return (
            "TRISC0 (Unpack) end-PC value.",
            "TRISC1 (Math) end-PC value.",
            "TRISC2 (Pack) end-PC value.",
        )
    if group == "Scratch":
        return (
            "TRISC0 (Unpack) scratch word.",
            "TRISC1 (Math) scratch word.",
            "TRISC2 (Pack) scratch word.",
        )

    raise RuntimeError(
        f"missing section-usage documentation for repeated descriptor {group}.{field.name}"
    )


def render_section_usages(usages: tuple[str, ...]) -> str:
    if not usages:
        return ""

    count = len(usages)
    labels = []
    for section, usage in enumerate(usages):
        # The current public Sec enum names S0 through S7.  Larger repeated
        # arrays still have physical sections, but do not have a named enumerator.
        enum_label = f" <code>Sec::S{section}</code>" if section <= 7 else ""
        labels.append(
            f"<li><strong>Section {section}</strong>{enum_label}: {esc(usage)}</li>"
        )
    range_label = (
        f"Sec::S0–Sec::S{count - 1}" if count <= 8 else f"sections 0–{count - 1}"
    )
    enum_limit = (
        '<p class="section-limit">The public <code>Sec</code> enum currently names only sections 0–7.</p>'
        if count > 8
        else ""
    )
    return (
        '<details class="section-usage">'
        f"<summary>Section usage ({range_label})</summary>"
        f"<ol>{''.join(labels)}</ol>{enum_limit}"
        "</details>"
    )


def catalog_group(field: FieldRecord, entry: str) -> tuple[str, str]:
    """Split a public expression into its navigable descriptor path and leaf."""
    owner = "::".join(field.owners)

    if field.family == "pack":
        address_ctrl = {
            "Pck0AddrCtrlXyReg0": "Packer[0].AddrCtrl[PackerReg::Reg0]",
            "Pck0AddrCtrlZwReg0": "Packer[0].AddrCtrl[PackerReg::Reg0]",
            "Pck0AddrCtrlXyReg1": "Packer[0].AddrCtrl[PackerReg::Reg1]",
            "Pck0AddrCtrlZwReg1": "Packer[0].AddrCtrl[PackerReg::Reg1]",
        }
        if owner in address_ctrl:
            return address_ctrl[owner], field.name
        address_base = {
            "Pck0AddrBaseReg0": "PackerReg::Reg0",
            "Pck0AddrBaseReg1": "PackerReg::Reg1",
        }
        if owner in address_base:
            return "Packer[0].AddrBase", f"[{address_base[owner]}]"
        if owner == "TileRowSetMappingRow::Fields":
            return "TileRowSetMapping[M]", "[S]"
        if owner == "TileFaceSetMappingRow::Fields":
            return "TileFaceSetMapping[M]", "[S]"

    if field.family == "unpack":
        if owner == "UnpackerAddrCtrlFields::Fields":
            return "Unpacker[U].AddrCtrl[R]", field.name
        if owner == "UnpackerAddrBaseFields::Fields":
            return "Unpacker[U].AddrBase", "[R]"
        if owner == "UnpackerBlobsYStartFields":
            return "Unpacker[0].BlobsYStart", f"[BlobContext::{field.name}]"
        if owner == "UnpackerContextDescriptor":
            return "Unpacker[U].Cntx[C]", "Base"
        if owner == "UnpackerFields::Fields":
            return "Unpacker[U]", field.name

    if field.family == "thcon":
        if owner == "ThconTileDescriptorFields":
            return "Thcon[Reg0].TileDescriptor", field.name
        match = re.fullmatch(r"ThconReg(\d+)Fields", owner)
        if match:
            return f"Thcon[Reg{match.group(1)}]", field.name

    if field.family == "thread":
        if owner == "DisableImpliedFmtFields":
            selector = {"SrcAField": "SrcA", "SrcBField": "SrcB"}[field.name]
            return "DisableImpliedFmt", f"[{selector}]"
        if owner == "AddrModFields::SrcFields":
            return "AddrMod[SrcA] / AddrMod[SrcB]", field.name
        if owner == "AddrModFields":
            return entry.rsplit(".", 1)[0], entry.rsplit(".", 1)[1]
        if owner == "PerfCntCmdEntry::Fields":
            return "PerfCntCmd[I]", field.name

    return owner, field.name


def location_html(field: FieldRecord) -> str:
    register_file, word_bits, base, word, shift, width, count, stride = field.args
    register_file = register_file.removeprefix("RegisterFile::")
    owner = "::".join(field.owners)
    if owner == "UnpackerContextDescriptor":
        register_file, word_bits = "State", "32"

    if owner == "UnpackerAddrCtrlFields::Fields":
        base = "44 + 12 * R + 2 * U" + (" + 1" if "+ 1" in base else "")
        word = "0"
    elif owner == "UnpackerAddrBaseFields::Fields":
        base, word = "48 + 12 * U + R", "0"
    elif owner == "UnpackerFields::Fields":
        base = base.replace("UnpackerIndex", "U")
    elif owner in {"TileRowSetMappingRow::Fields", "TileFaceSetMappingRow::Fields"}:
        base = base.replace("MappingIndex", "M")
        shift = shift.replace("SetIndex", "S")

    if owner == "UnpackerContextDescriptor":
        return (
            '<span class="file state">State</span>'
            "<code>U=0: word 76 + C</code><code>U=1: word 124 + C</code><code>bits 31:0</code>"
        )

    if owner == "AddrModFields::SrcFields":
        layouts = {
            "Incr": ("word 12", "SrcA bits 5:0; SrcB bits 13:8"),
            "Incr2": ("word 20", "SrcA bit 0; SrcB bit 1"),
            "CR": ("word 12", "SrcA bit 6; SrcB bit 14"),
            "Clear": ("word 12", "SrcA bit 7; SrcB bit 15"),
        }
        address, bits = layouts[field.name]
        return (
            '<span class="file thread">Thread</span>'
            f"<code>{esc(address)}</code><code>{esc(bits)}</code><small>{esc(section_note(field))}</small>"
        )

    if base.isdigit() and word.isdigit():
        address = str(int(base) + int(word))
    elif word == "0":
        address = base
    else:
        address = f"{base} + {word}"

    if shift.isdigit() and width.isdigit():
        bit = int(shift)
        bit_width = int(width)
        if bit_width == 1:
            bits = f"bit {bit}"
        elif bit_width <= int(word_bits):
            bits = f"bits {bit + bit_width - 1}:{bit}"
        else:
            bits = f"{bit_width} bits from bit {bit}"
    else:
        bits = f"shift {shift}; width {width}"

    repeated = ""
    if count != "1":
        repeated = f"<small>{esc(section_note(field))}</small>"
    return (
        f'<span class="file {esc(register_file.lower())}">{esc(register_file)}</span>'
        f"<code>word {esc(address)}</code><code>{esc(bits)}</code>{repeated}"
    )


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def descriptor_suggestion_id(family: str, group: str) -> str:
    return f"descriptor:{family}:{group}"


def field_suggestion_id(field: FieldRecord) -> str:
    source_path = field.source.relative_to(ROOT).as_posix()
    return f"field:{field.family}:{source_path}:{field.declaration}"


def render_field_row(
    field: FieldRecord, group: str, member: str, entry: str, indices: str
) -> str:
    source_path = field.source.relative_to(ROOT).as_posix()
    description = (
        esc(field.comment)
        if field.comment is not None
        else '<span class="missing">No field comment in source.</span>'
    )
    usages = section_usages(field, group)
    search = " ".join(
        (
            field.family,
            group,
            member,
            entry,
            field.declaration,
            indices,
            field.comment or "",
            *usages,
        )
    )
    suggestion_id = field_suggestion_id(field)
    return f"""              <tr class="field-row" data-search="{esc(search.lower())}" data-family="{esc(field.family)}">
          <td class="field-name"><code>{esc(member)}</code><small><code>{esc(entry)}</code></small></td>
          <td>{description}</td>
          <td class="location">{location_html(field)}</td>
          <td class="source"><a href="../../{esc(source_path)}">{esc(field.source.name)}:{field.line}</a></td>
          <td class="suggestion-cell">
            <button class="suggest-name" type="button" data-suggestion-type="field" data-target-id="{esc(suggestion_id)}" data-family="{esc(field.family)}" data-descriptor="{esc(group)}" data-field="{esc(member)}" data-current-name="{esc(member)}">Suggest field name</button>
            <div class="suggestion-list" data-suggestions-for="{esc(suggestion_id)}"></div>
          </td>
        </tr>"""


def render_family(family: str, records: list[FieldRecord]) -> str:
    groups: dict[str, list[tuple[FieldRecord, str, str, str]]] = {}
    for field in records:
        entry, indices = public_entry_point(field)
        group, member = catalog_group(field, entry)
        groups.setdefault(group, []).append((field, member, entry, indices))

    rendered_groups: list[str] = []
    for group, fields in groups.items():
        suggestion_id = descriptor_suggestion_id(family, group)
        usage_sets = {section_usages(field, group) for field, _, _, _ in fields}
        if len(usage_sets) != 1:
            raise RuntimeError(
                f"fields in descriptor {group} disagree about section usage"
            )
        usages = usage_sets.pop()
        usage_html = render_section_usages(usages)
        rows = "\n".join(
            render_field_row(field, group, member, entry, indices)
            for field, member, entry, indices in fields
        )
        ranges = list(dict.fromkeys(indices for _, _, _, indices in fields if indices))
        range_html = (
            f'<small class="range">{" · ".join(esc(value) for value in ranges)}</small>'
            if ranges
            else ""
        )
        search = " ".join(
            [family, group]
            + [
                " ".join(
                    (member, entry, field.declaration, indices, field.comment or "")
                )
                for field, member, entry, indices in fields
            ]
        )
        rendered_groups.append(
            f"""        <article class="field-group" data-search="{esc(search.lower())}">
          <header class="group-head">
            <div><span>Descriptor path</span><code>{esc(group)}</code>{range_html}</div>
            <div class="group-suggestion">
              <span class="group-count">{len(fields)} {"field" if len(fields) == 1 else "fields"}</span>
              <button class="suggest-name" type="button" data-suggestion-type="descriptor" data-target-id="{esc(suggestion_id)}" data-family="{esc(family)}" data-descriptor="{esc(group)}" data-field="" data-current-name="{esc(group)}">Suggest descriptor name</button>
              <div class="suggestion-list" data-suggestions-for="{esc(suggestion_id)}"></div>
            </div>
          </header>
          {usage_html}
          <div class="table-wrap">
            <table>
              <thead><tr><th>Field</th><th>Exact source description</th><th>Physical location</th><th>Defined in</th><th>Naming suggestions</th></tr></thead>
              <tbody>
{rows}
              </tbody>
            </table>
          </div>
        </article>"""
        )

    return f"""      <section class="family-section" id="family-{esc(family)}" data-family="{esc(family)}">
        <div class="family-head"><h2>{esc(family.upper())}</h2><span>{len(records)} catalog entries · {len(groups)} descriptor paths</span></div>
{chr(10).join(rendered_groups)}
      </section>"""


def render(records: list[FieldRecord]) -> str:
    counts = {
        family: sum(record.family == family for record in records)
        for family in FAMILIES
    }
    filter_buttons = "\n".join(
        f'          <button type="button" data-family="{family}">{family.upper()} <span>{counts[family]}</span></button>'
        for family in FAMILIES
    )
    family_sections = "\n".join(
        render_family(family, [record for record in records if record.family == family])
        for family in FAMILIES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Searchable catalog of every Blackhole hal::cfg Field descriptor and its public C++ entry point.">
  <title>Field catalog — Blackhole hal::cfg</title>
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <script src="hal_theme.js"></script>
  <link rel="stylesheet" href="hal_docs.css">
  <script src="hal_docs.js" defer></script>
  <style>
    :root {{ color-scheme: light; --bg:#fff; --subtle:#f6f8fa; --border:#d8dee4; --text:#1f2328; --muted:#59636e; --link:#0969da; --accent:#007a6f; --accent-soft:#e8f7f4; --amber:#9a6700; --amber-soft:#fff8c5; --red:#cf222e; --red-soft:#ffebe9; --sidebar:248px; --topbar:58px; --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace; --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--text); background:var(--bg); font:15px/1.55 var(--sans); }}
    a {{ color:var(--link); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    code {{ font-family:var(--mono); }}
    .topbar {{ position:sticky; top:0; z-index:20; display:flex; align-items:center; gap:12px; min-height:var(--topbar); padding:10px 24px; border-bottom:1px solid var(--border); background:rgba(255,255,255,.97); }}
    .topbar .brand {{ color:var(--text); font-weight:700; }} .crumb {{ color:#818b98; }} .topbar code {{ padding:3px 7px; border-radius:5px; color:var(--accent); background:var(--accent-soft); font-weight:650; }}
    .layout {{ display:grid; grid-template-columns:var(--sidebar) minmax(0,1380px); justify-content:center; min-height:calc(100vh - var(--topbar)); }}
    .sidebar {{ position:sticky; top:var(--topbar); align-self:start; max-height:calc(100vh - var(--topbar)); padding:26px 20px 40px; overflow-y:auto; border-right:1px solid var(--border); }}
    .sidebar h2 {{ margin:0 0 9px; color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
    .sidebar h2:not(:first-child) {{ margin-top:23px; }}
    .sidebar a {{ display:block; padding:4px 8px; border-radius:5px; color:var(--muted); font-size:13px; }}
    .sidebar a:hover {{ color:var(--link); background:var(--subtle); text-decoration:none; }}
    .sidebar .current {{ color:var(--accent); background:var(--accent-soft); font-weight:650; }}
    main {{ min-width:0; padding:42px 32px 80px; }}
    .package {{ color:var(--accent); font:650 13px/1 var(--mono); }}
    h1 {{ margin:12px 0 8px; font-size:clamp(2rem,5vw,3rem); letter-spacing:-.035em; }}
    .lede {{ max-width:82ch; margin:0; color:var(--muted); font-size:18px; }}
    .note {{ max-width:100ch; margin:22px 0; padding:14px 17px; border:1px solid var(--border); border-left:4px solid var(--accent); border-radius:7px; background:var(--accent-soft); }}
    #suggestion-service-status.error {{ color:#cf222e; font-weight:650; }}
    .controls {{ position:sticky; top:var(--topbar); z-index:15; display:flex; flex-wrap:wrap; gap:9px; margin:28px 0 16px; padding:14px; border:1px solid var(--border); border-radius:9px; background:rgba(255,255,255,.97); box-shadow:0 4px 16px rgba(31,35,40,.06); }}
    .controls > input {{ flex:1 1 340px; min-width:220px; padding:9px 11px; border:1px solid #b6bec8; border-radius:6px; font:14px var(--sans); }}
    button {{ padding:8px 10px; border:1px solid var(--border); border-radius:6px; color:var(--muted); background:var(--subtle); cursor:pointer; font-weight:650; }} button span {{ color:#818b98; font-weight:400; }} button.active {{ color:var(--accent); border-color:#8bd3ca; background:var(--accent-soft); }}
    .result-count {{ align-self:center; min-width:98px; color:var(--muted); text-align:right; font-size:13px; }}
    .family-section {{ margin-top:44px; }}
    .family-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:18px; margin-bottom:15px; border-bottom:1px solid var(--border); }}
    .family-head h2 {{ margin:0; padding:0 0 8px; font-size:1.6rem; letter-spacing:-.02em; }} .family-head span {{ color:var(--muted); font-size:12px; }}
    .field-group {{ margin:0 0 18px; border:1px solid var(--border); border-radius:9px; overflow:hidden; background:var(--bg); }}
    .group-head {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:13px 15px; border-bottom:1px solid var(--border); background:var(--subtle); }}
    .group-head div {{ min-width:0; }} .group-head > div:first-child > span {{ display:block; margin-bottom:4px; color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
    .group-head code {{ color:var(--accent); font-size:14px; font-weight:650; overflow-wrap:anywhere; }} .group-head .range {{ margin-top:4px; color:var(--muted); font-size:11px; }} .group-count {{ flex:none; color:var(--muted); font-size:11px; }}
    .group-suggestion {{ display:flex; flex:none; flex-wrap:wrap; align-items:center; justify-content:flex-end; gap:8px; max-width:45%; }}
    .group-suggestion .suggestion-list {{ flex-basis:100%; text-align:right; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; min-width:1120px; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
    th,td {{ padding:10px 11px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); background:#fbfcfd; font-size:10px; letter-spacing:.04em; text-transform:uppercase; }}
    tbody tr:hover {{ background:#fbfcfd; }} tbody tr:last-child td {{ border-bottom:0; }}
    th:nth-child(1) {{ width:20%; }} th:nth-child(2) {{ width:30%; }} th:nth-child(3) {{ width:15%; }} th:nth-child(4) {{ width:14%; }} th:nth-child(5) {{ width:21%; }}
    td code {{ overflow-wrap:anywhere; }} small {{ display:block; margin-top:4px; color:var(--muted); font-size:11px; }}
    .field-name > code {{ color:var(--text); font-size:13px; font-weight:700; }} .field-name small code, .source small code {{ color:var(--muted); font-weight:400; }}
    .location {{ white-space:normal; }} .location > code {{ display:block; margin-top:5px; color:var(--muted); font-size:11px; }}
    .file {{ display:inline-block; padding:2px 7px; border-radius:999px; font:650 10px/1.5 var(--mono); text-transform:uppercase; }} .file.state {{ color:#0550ae; background:#eef5ff; }} .file.thread {{ color:#9a6700; background:#fff8c5; }}
    .missing {{ color:#818b98; font-style:italic; }} .empty {{ display:none; margin:24px 0; color:var(--muted); text-align:center; }}
    .section-usage {{ margin:0; padding:10px 15px; border-bottom:1px solid var(--border); color:var(--muted); background:var(--accent-soft); }}
    .section-usage summary {{ color:var(--accent); cursor:pointer; font-size:12px; font-weight:700; }}
    .section-usage ol {{ margin:7px 0 0; padding-left:22px; }}
    .section-usage li + li {{ margin-top:5px; }}
    .section-usage strong {{ color:var(--text); }}
    .section-usage code {{ color:var(--muted); font-size:11px; }}
    .section-limit {{ margin:7px 0 0; font-size:11px; }}
    .suggest-name {{ padding:6px 8px; color:var(--accent); border-color:#8bd3ca; background:var(--accent-soft); font-size:11px; white-space:nowrap; }}
    .suggest-name:disabled {{ cursor:not-allowed; opacity:.6; }}
    .suggestion-list:empty {{ display:none; }}
    .suggestion-history {{ margin-top:7px; color:var(--muted); font-size:11px; }}
    .suggestion-history summary {{ cursor:pointer; font-weight:650; }}
    .suggestion-history ul {{ margin:7px 0 0; padding-left:17px; }}
    .suggestion-history li + li {{ margin-top:6px; }}
    .suggestion-history code {{ color:var(--text); font-weight:650; }}
    .suggestion-history small {{ margin-top:1px; }}
    dialog {{ width:min(560px,calc(100vw - 28px)); padding:0; border:1px solid var(--border); border-radius:10px; color:var(--text); background:var(--bg); box-shadow:0 20px 60px rgba(31,35,40,.28); }}
    dialog::backdrop {{ background:rgba(31,35,40,.45); }}
    .suggestion-form {{ padding:22px; }}
    .suggestion-form h2 {{ margin:0 0 4px; font-size:1.35rem; }}
    .suggestion-target {{ margin:0 0 18px; color:var(--muted); overflow-wrap:anywhere; }}
    .suggestion-form label {{ display:block; margin-top:13px; color:var(--muted); font-size:12px; font-weight:650; }}
    .suggestion-form input,.suggestion-form textarea {{ display:block; width:100%; margin-top:5px; padding:9px 10px; border:1px solid #b6bec8; border-radius:6px; color:var(--text); background:var(--bg); font:14px var(--sans); }}
    .suggestion-form textarea {{ min-height:92px; resize:vertical; }}
    .suggestion-status {{ min-height:20px; margin:10px 0 0; color:var(--muted); font-size:12px; }}
    .suggestion-status.error {{ color:#cf222e; }}
    .dialog-actions {{ display:flex; justify-content:flex-end; gap:8px; margin-top:18px; }}
    footer {{ margin-top:28px; color:var(--muted); font-size:12px; }}
    @media (max-width:820px) {{ .topbar {{ padding:10px 14px; }} .layout {{ display:block; }} .sidebar {{ position:static; max-height:none; padding:14px 16px; overflow:visible; border-right:0; border-bottom:1px solid var(--border); }} .sidebar h2 {{ display:none; }} .sidebar nav {{ display:flex; gap:5px; overflow-x:auto; }} .sidebar a {{ white-space:nowrap; }} main {{ padding:28px 14px 60px; }} .controls {{ top:var(--topbar); }} .result-count {{ width:100%; text-align:left; }} .family-head {{ display:block; padding-bottom:8px; }} .group-head {{ align-items:flex-start; }} .group-suggestion {{ flex-direction:column; align-items:flex-end; }} }}
    @media print {{ .topbar,.sidebar {{ display:none; }} .layout {{ display:block; }} main {{ padding:0; }} }}
  </style>
</head>
<body>
  <!-- Generated by docs/hal/gen_cfg_field_catalog.py. Do not edit this file directly. -->
  <header class="topbar">
    <a class="brand" href="index.html">TT-LLK</a><span class="crumb">/</span><span>Blackhole</span><span class="crumb">/</span><code>hal::cfg</code><span class="crumb">/</span><span>Fields</span>
  </header>
  <div class="layout">
    <aside class="sidebar" aria-label="Documentation navigation">
      <nav>
        <h2>Packages</h2>
        <a href="index.html">Documentation home</a>
        <a href="cfg_hal_interface.html">hal::cfg</a>
        <a class="current" href="cfg_field_catalog.html">hal::cfg field catalog</a>
        <a href="address_counters_hal_interface.html">hal::address_counters</a>
        <a href="gpr_ops_hal_interface.html">hal::gpr_ops</a>
        <a href="atomic_hal_interface.html">hal::atomic</a>
        <a href="mop_hal_interface.html">hal::mop</a>
        <a href="replay_hal_interface.html">hal::replay</a>
        <a href="sync_hal_interface.html">hal::sync</a>
        <a href="unpack_hal_interface.html">hal::unpack</a>
        <a href="fpu_hal_interface.html">hal::fpu</a>

        <h2>Catalog</h2>
        <a href="#overview">Overview</a>
        <a href="#filters">Search and filters</a>
        <a href="#family-thread">Thread</a>
        <a href="#family-alu">ALU</a>
        <a href="#family-pack">Pack</a>
        <a href="#family-unpack">Unpack</a>
        <a href="#family-thcon">THCON</a>
        <a href="#family-global">Global</a>
      </nav>
    </aside>

    <main id="overview">
    <div class="package">Blackhole / namespace hal::cfg</div>
    <h1>Field catalog</h1>
    <p class="lede">Every Blackhole CFG field, organized by the public descriptor hierarchy used in C++.</p>
    <div class="note">
      <strong>The public hierarchy comes first.</strong> Fields are grouped under usable paths such as <code>Packer[0].AddrCtrl[PackerReg::Reg0]</code>, <code>Unpacker[U].Cntx[C]</code>, and <code>Thcon[Reg2]</code>. Generated storage types are treated as implementation details. Descriptions are copied verbatim from field comments; missing comments are identified instead of replaced with generated prose.
    </div>
    <div class="note">
      <strong>Help improve the names.</strong> Every descriptor and field has its own suggestion control. Suggestions are shared and persist across catalog updates.
      <small id="suggestion-service-status">Connecting to the suggestion store…</small>
    </div>
    <div class="controls" id="filters" aria-label="Field catalog filters">
      <input id="search" type="search" placeholder="Search entry point, field name, or description…" autocomplete="off">
      <button class="active" type="button" data-family="all">ALL <span>{len(records)}</span></button>
{filter_buttons}
      <span class="result-count" id="result-count">{len(records)} entries</span>
    </div>
    <div id="field-groups">
{family_sections}
    </div>
    <p class="empty" id="empty">No fields match the current filters.</p>
    <footer>Generated from {len(records)} <code>static constexpr Field</code> declarations by <code>docs/hal/gen_cfg_field_catalog.py</code>. Naming suggestions are stored separately and are never overwritten by generation.</footer>
    </main>
  </div>
  <dialog id="suggestion-dialog">
    <form class="suggestion-form" id="suggestion-form">
      <h2 id="suggestion-title">Suggest a name</h2>
      <p class="suggestion-target" id="suggestion-target"></p>
      <label>Suggested name
        <input id="suggested-name" name="suggested_name" maxlength="200" required autocomplete="off">
      </label>
      <label>Your name (optional)
        <input id="suggested-by" name="suggested_by" maxlength="100" autocomplete="name">
      </label>
      <label>Reasoning or context (optional)
        <textarea id="suggestion-rationale" name="rationale" maxlength="1000"></textarea>
      </label>
      <p class="suggestion-status" id="suggestion-status" role="status"></p>
      <div class="dialog-actions">
        <button id="suggestion-cancel" type="button">Cancel</button>
        <button class="suggest-name" id="suggestion-submit" type="submit">Save suggestion</button>
      </div>
    </form>
  </dialog>
  <script>
    const search = document.getElementById('search');
    const sections = [...document.querySelectorAll('.family-section')];
    const rows = [...document.querySelectorAll('.field-row')];
    const buttons = [...document.querySelectorAll('button[data-family]')];
    const count = document.getElementById('result-count');
    const empty = document.getElementById('empty');
    const suggestionButtons = [...document.querySelectorAll('.suggest-name[data-target-id]')];
    const suggestionDialog = document.getElementById('suggestion-dialog');
    const suggestionForm = document.getElementById('suggestion-form');
    const suggestionTitle = document.getElementById('suggestion-title');
    const suggestionTarget = document.getElementById('suggestion-target');
    const suggestedName = document.getElementById('suggested-name');
    const suggestedBy = document.getElementById('suggested-by');
    const suggestionRationale = document.getElementById('suggestion-rationale');
    const suggestionStatus = document.getElementById('suggestion-status');
    const suggestionSubmit = document.getElementById('suggestion-submit');
    const suggestionServiceStatus = document.getElementById('suggestion-service-status');
    let activeSuggestionButton = null;
    let savedSuggestions = [];
    const requestedFamily = new URLSearchParams(window.location.search).get('family');
    let family = buttons.some(button => button.dataset.family === requestedFamily) ? requestedFamily : 'all';
    const requestedSearch = new URLSearchParams(window.location.search).get('q');
    if (requestedSearch) search.value = requestedSearch;

    function renderSuggestions() {{
      const byTarget = new Map();
      for (const suggestion of savedSuggestions) {{
        if (!byTarget.has(suggestion.target_id)) byTarget.set(suggestion.target_id, []);
        byTarget.get(suggestion.target_id).push(suggestion);
      }}
      for (const container of document.querySelectorAll('[data-suggestions-for]')) {{
        container.replaceChildren();
        const suggestions = byTarget.get(container.dataset.suggestionsFor) || [];
        if (!suggestions.length) continue;
        const details = document.createElement('details');
        details.className = 'suggestion-history';
        const summary = document.createElement('summary');
        summary.textContent = `${{suggestions.length}} suggestion${{suggestions.length === 1 ? '' : 's'}}`;
        const list = document.createElement('ul');
        for (const suggestion of suggestions) {{
          const item = document.createElement('li');
          const name = document.createElement('code');
          name.textContent = suggestion.suggested_name;
          item.appendChild(name);
          if (suggestion.rationale) item.appendChild(document.createTextNode(` — ${{suggestion.rationale}}`));
          const metadata = document.createElement('small');
          const author = suggestion.suggested_by ? `by ${{suggestion.suggested_by}} · ` : '';
          const created = new Date(suggestion.created_at);
          metadata.textContent = author + (Number.isNaN(created.getTime()) ? suggestion.created_at : created.toLocaleString());
          item.appendChild(metadata);
          list.appendChild(item);
        }}
        details.append(summary, list);
        container.appendChild(details);
      }}
      for (const button of suggestionButtons) {{
        const countForTarget = (byTarget.get(button.dataset.targetId) || []).length;
        button.textContent = countForTarget ? 'Suggest another name' :
          (button.dataset.suggestionType === 'descriptor' ? 'Suggest descriptor name' : 'Suggest field name');
      }}
    }}

    function updateSuggestionServiceStatus() {{
      const total = savedSuggestions.length;
      suggestionServiceStatus.classList.remove('error');
      suggestionServiceStatus.textContent = `${{total}} shared naming suggestion${{total === 1 ? '' : 's'}} loaded.`;
    }}

    async function suggestionResponseJson(response) {{
      const body = await response.text();
      try {{
        return JSON.parse(body);
      }} catch (error) {{
        const contentType = response.headers.get('Content-Type') || 'unknown content type';
        throw new Error(`Suggestion API ${{response.url}} returned HTTP ${{response.status}} with ${{contentType}}, not JSON`);
      }}
    }}

    async function loadSuggestions() {{
      try {{
        const response = await fetch('../../__suggestions__', {{ cache: 'no-store' }});
        const payload = await suggestionResponseJson(response);
        if (!response.ok || !Array.isArray(payload.suggestions)) throw new Error(payload.message || 'Invalid suggestion response');
        savedSuggestions = payload.suggestions;
        renderSuggestions();
        updateSuggestionServiceStatus();
      }} catch (error) {{
        suggestionServiceStatus.classList.add('error');
        const endpoint = new URL('../../__suggestions__', window.location.href).href;
        suggestionServiceStatus.textContent = `Shared suggestion service unavailable at ${{endpoint}}: ${{error.message}} No browser-local data will be used.`;
      }}
    }}

    for (const button of suggestionButtons) {{
      button.addEventListener('click', () => {{
        activeSuggestionButton = button;
        suggestionForm.reset();
        suggestionStatus.textContent = '';
        suggestionStatus.classList.remove('error');
        const kind = button.dataset.suggestionType;
        suggestionTitle.textContent = kind === 'descriptor' ? 'Suggest a descriptor name' : 'Suggest a field name';
        suggestionTarget.textContent = `Current ${{kind}}: ${{button.dataset.currentName}}`;
        suggestionDialog.showModal();
        suggestedName.focus();
      }});
    }}

    document.getElementById('suggestion-cancel').addEventListener('click', () => suggestionDialog.close());
    suggestionDialog.addEventListener('click', event => {{
      if (event.target === suggestionDialog) suggestionDialog.close();
    }});
    suggestionForm.addEventListener('submit', async event => {{
      event.preventDefault();
      if (!activeSuggestionButton) return;
      suggestionSubmit.disabled = true;
      suggestionStatus.classList.remove('error');
      suggestionStatus.textContent = 'Saving…';
      const payload = {{
        target_type: activeSuggestionButton.dataset.suggestionType,
        target_id: activeSuggestionButton.dataset.targetId,
        family: activeSuggestionButton.dataset.family,
        descriptor: activeSuggestionButton.dataset.descriptor,
        field: activeSuggestionButton.dataset.field,
        current_name: activeSuggestionButton.dataset.currentName,
        suggested_name: suggestedName.value,
        suggested_by: suggestedBy.value,
        rationale: suggestionRationale.value
      }};
      try {{
        const response = await fetch('../../__suggestions__', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const result = await suggestionResponseJson(response);
        if (!response.ok) throw new Error(result.message || 'Could not save suggestion');
        savedSuggestions.push(result.suggestion);
        renderSuggestions();
        updateSuggestionServiceStatus();
        suggestionDialog.close();
      }} catch (error) {{
        suggestionStatus.textContent = `Shared save failed: ${{error.message}}. Confirm the site is running serve_hal_docs.py.`;
        suggestionStatus.classList.add('error');
      }} finally {{
        suggestionSubmit.disabled = false;
      }}
    }});

    function update() {{
      const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
      let visible = 0;
      for (const section of sections) {{
        const familyMatch = family === 'all' || section.dataset.family === family;
        let sectionVisible = 0;
        for (const group of section.querySelectorAll('.field-group')) {{
          let groupVisible = 0;
          for (const row of group.querySelectorAll('.field-row')) {{
            const textMatch = terms.every(term => row.dataset.search.includes(term));
            row.hidden = !(familyMatch && textMatch);
            if (!row.hidden) {{ groupVisible++; sectionVisible++; visible++; }}
          }}
          group.hidden = groupVisible === 0;
        }}
        section.hidden = sectionVisible === 0;
      }}
      count.textContent = `${{visible}} entr${{visible === 1 ? 'y' : 'ies'}}`;
      empty.style.display = visible ? 'none' : 'block';
    }}
    search.addEventListener('input', update);
    for (const button of buttons) {{
      button.classList.toggle('active', button.dataset.family === family);
      button.addEventListener('click', () => {{
        family = button.dataset.family;
        for (const candidate of buttons) candidate.classList.toggle('active', candidate === button);
        update();
      }});
    }}
    loadSuggestions();
    update();
  </script>
</body>
</html>
"""


def main() -> None:
    records = [record for family in FAMILIES for record in parse_header(family)]
    OUTPUT.write_text(render(records), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} fields")


if __name__ == "__main__":
    main()
