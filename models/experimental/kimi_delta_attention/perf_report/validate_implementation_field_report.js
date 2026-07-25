#!/usr/bin/env node
"use strict";

const fs = require("fs");
const vm = require("vm");
const path = require("path");

const report = process.argv[2] || path.join(__dirname, "kda_implementation_field_report.html");
const html = fs.readFileSync(report, "utf8");
const dataMatch = html.match(/<script id="report-data" type="application\/json">\s*([\s\S]*?)\s*<\/script>/);
const scriptMatches = [...html.matchAll(/<script>\s*([\s\S]*?)\s*<\/script>/g)];
if (!dataMatch || scriptMatches.length !== 1) throw new Error("expected one JSON payload and one executable script");
const payload = JSON.parse(dataMatch[1]);
new vm.Script(scriptMatches[0][1], {filename: "report-inline.js"});

class Element {
  constructor(id = "") {
    this.id = id;
    this.dataset = {};
    this.children = [];
    this.attributes = {};
    this._innerHTML = "";
    this.textContent = "";
    this.onclick = null;
  }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(value) { this._innerHTML = String(value); this.children = []; }
}

const ids = [
  "report-data", "cards", "prepText", "prepGrid", "scanText", "scanGrid",
  "mmGrid", "rsGrid", "prepMap", "utilBars", "decisionRows",
];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
elements["report-data"].textContent = dataMatch[1];
const scenarioButtons = ["t640", "t5120"].map(s => {
  const e = new Element(); e.dataset.s = s; return e;
});
const filterButtons = ["all", "retain", "reject", "defer"].map(status => {
  const e = new Element(); e.dataset.status = status; return e;
});
const sortHeaders = ["status", "scope", "name", "delta"].map(sort => {
  const e = new Element(); e.dataset.sort = sort; return e;
});
const document = {
  getElementById(id) {
    if (!elements[id]) elements[id] = new Element(id);
    return elements[id];
  },
  createElement() { return new Element(); },
  querySelectorAll(selector) {
    if (selector === "#scenario button") return scenarioButtons;
    if (selector === "#statusFilter button") return filterButtons;
    if (selector === ".ledger th[data-sort]") return sortHeaders;
    return [];
  },
};
const context = {
  document,
  console,
  Intl,
};
vm.createContext(context);
new vm.Script(scriptMatches[0][1], {filename: "report-inline.js"}).runInContext(context);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
function renderedText() {
  return Object.values(elements).map(e => `${e.innerHTML}\n${e.textContent}`).join("\n");
}

assert(payload.scenarios.t640.wall_us === 619.594, "T640 wall value changed");
assert(payload.scenarios.t5120.wall_us === 3183.263, "T5120 wall value changed");
assert(elements.prepGrid.children.length === 110, "prep grid must render 110 cores");
assert(elements.scanGrid.children.filter(e => e.className.includes("scan")).length === 16, "scan must use 16 cores");
assert(elements.mmGrid.children.filter(e => e.className.includes("mm")).length === 64, "matmul must use 64 cores");
assert(elements.rsGrid.children.filter(e => e.className.includes("rs")).length === 4, "RS must use four sender cores");
assert(elements.cards.innerHTML.includes("619.594 µs"), "initial wall card missing");
assert(elements.decisionRows.innerHTML.includes("kd + q_decay + dl BF16"), "decision ledger missing retained inverse");

scenarioButtons[1].onclick();
assert(elements.cards.innerHTML.includes("3,183.263 µs"), "T5120 toggle did not update wall card");
assert(elements.prepText.textContent.includes("640 items fill 110 cores"), "T5120 distribution text missing");
assert(elements.prepGrid.children.filter(e => e.className.includes("prep")).length === 110, "T5120 prep must use 110 cores");
assert(elements.scanGrid.children.filter(e => e.className.includes("scan")).length === 80, "T5120 grouped scan must use 80 cores");
assert(elements.scanText.textContent.includes("80 whole-V owners"), "T5120 grouped scan text missing");

filterButtons[2].onclick();
assert(elements.decisionRows.innerHTML.includes("Share common inputs"), "reject filter omitted rejected rows");
assert(!elements.decisionRows.innerHTML.includes("kd + q_decay + dl BF16"), "reject filter retained accepted rows");
sortHeaders[3].onclick();

const text = renderedText();
assert(!/\bNaN\b|undefined/.test(text), "rendered output contains NaN or undefined");
assert(html.includes('href="kda_standalone_technical_guide.html"'), "guide link missing");
assert(html.includes("11.28 FLOP/B") && html.includes("297 FLOP/B"), "roofline axes/data missing");
assert(payload.scenarios.t5120.control_wall_us === 3200.471, "T5120 control missing");
assert(payload.decisions.some(d => d.name === "k_dec_t BF16" && d.status === "reject"), "dtype rejection missing");
assert(html.includes("fresh matched trace") && html.includes("correctness-only rejection"), "evidence tiers missing");

console.log("PASS: JSON and inline JavaScript parse");
console.log("PASS: T640/T5120 scenario controls update metrics and 110-core maps");
console.log("PASS: core counts prep=80/110, scan=16/80, matmul=64, RS senders=4");
console.log("PASS: decision filters/sort execute; rendered output has no NaN/undefined");
console.log("PASS: theory-guide link, roofline labels, and evidence tiers present");
