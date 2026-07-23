// Headless interaction and data-integrity harness for kda_perf_report.html.
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const report = fs.readFileSync(path.join(__dirname, "kda_perf_report.html"), "utf8");
const payloadMatch = report.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/);
const scripts = [...report.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
if (!payloadMatch || scripts.length < 2) throw new Error("report scripts not found");
const payload = payloadMatch[1];
let app = scripts[scripts.length - 1][1];

const renders = [];
function element(id = "") {
  const classes = new Set();
  const el = {
    id, dataset: {}, attrs: {}, style: {}, children: [], _html: "", _text: "",
    onclick: null, onmousedown: null,
    classList: {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      contains: name => classes.has(name),
    },
    setAttribute(key, value) { this.attrs[key] = String(value); },
    getAttribute(key) { return this.attrs[key] ?? null; },
    querySelectorAll() { return []; },
  };
  Object.defineProperty(el, "innerHTML", {
    get() { return this._html; },
    set(value) { this._html = String(value); renders.push(this._html); },
  });
  Object.defineProperty(el, "textContent", {
    get() { return this._text; },
    set(value) { this._text = String(value); renders.push(this._text); },
  });
  return el;
}

const ids = {};
const scenarios = ["t640", "t5120"].map(value => {
  const button = element(); button.dataset.s = value; return button;
});
const views = ["semantic", "ops"].map(value => {
  const button = element(); button.dataset.v = value; return button;
});
const zooms = ["in", "out", "reset"].map(value => {
  const button = element(); button.dataset.z = value; return button;
});
const sorts = ["order", "label", "block", "duration_ns", "p10_ns", "p90_ns", "cores"].map(value => {
  const button = element(); button.dataset.sort = value; return button;
});

global.document = {
  getElementById(id) {
    if (!ids[id]) ids[id] = element(id);
    return ids[id];
  },
  querySelectorAll(selector) {
    if (selector === "#scenarioSeg button") return scenarios;
    if (selector === "#viewSeg button") return views;
    if (selector === ".zoom button") return zooms;
    if (selector === "th[data-sort]") return sorts;
    return [];
  },
};
document.getElementById("payload")._text = payload;
global.window = global;
app += "\n;globalThis.__REPORT_API__=globalThis.__KDA_REPORT__;";

const failures = [];
function check(name, action) {
  try { action(); console.log("ok  " + name); }
  catch (error) { failures.push(name + ": " + error.message); console.log("FAIL " + name + ": " + error.message); }
}
function click(button) {
  if (typeof button.onclick !== "function") throw new Error("missing click handler");
  button.onclick({ clientX: 0, clientY: 0 });
}

check("load and initial render", () => vm.runInThisContext(app));
const api = global.__REPORT_API__;
const P = JSON.parse(payload);

check("payload scenarios and graph structure", () => {
  if (Object.keys(P.scenarios).join(",") !== "t640,t5120") throw new Error("scenario keys");
  if (P.blocks.length !== 9 || P.edges.length < 9) throw new Error("semantic graph incomplete");
});
for (const key of ["t640", "t5120"]) {
  check(key + " trace invariants", () => {
    const scenario = P.scenarios[key];
    if (scenario.replay_spans_ns.length !== 10) throw new Error("not ten replays");
    if (scenario.calls.length !== (key === "t640" ? 30 : 35)) throw new Error("call count");
    const callSum = scenario.calls.reduce((sum, call) => sum + call.duration_ns, 0);
    const blockSum = Object.values(scenario.block_timing_ns).reduce((sum, value) => sum + value, 0);
    if (Math.round(callSum) !== Math.round(blockSum) || Math.round(blockSum) !== Math.round(scenario.graph_total_ns)) {
      throw new Error("active attribution does not close");
    }
    if (!(scenario.latency_ns > 0 && scenario.compute_util > 0 && scenario.ccl_util > 0)) throw new Error("non-positive metric");
  });
  check(key + " toggle and render", () => click(scenarios.find(button => button.dataset.s === key)));
  check(key + " semantic graph", () => {
    api.setView("semantic");
    if (!document.getElementById("graph").innerHTML.includes('class="node"')) throw new Error("no semantic nodes");
  });
  check(key + " operation graph", () => {
    api.setView("ops");
    if (!document.getElementById("graph").innerHTML.includes('class="opnode"')) throw new Error("no operation nodes");
  });
}
check("node expand", () => {
  api.setScenario("t640"); api.setView("semantic"); api.expanded.add("convolution"); api.drawGraph();
  if (!document.getElementById("graph").innerHTML.includes('class="opnode"')) throw new Error("expanded calls absent");
});
check("source drawer", () => {
  api.openDrawer("recurrence");
  const body = document.getElementById("drawerBody").innerHTML;
  if (!/chunk_gdn_phased\.cpp:\d+-\d+/.test(body) || !body.includes("compute_output_specs")) throw new Error("source proof absent");
});
check("sortable operation table", () => {
  const before = document.getElementById("opRows").innerHTML;
  click(sorts.find(button => button.dataset.sort === "duration_ns"));
  const after = document.getElementById("opRows").innerHTML;
  if (before === after) throw new Error("sort did not change rows");
});
check("zoom and scenario reset", () => {
  const initial = api.state().scale;
  click(zooms.find(button => button.dataset.z === "in"));
  if (!(api.state().scale > initial)) throw new Error("zoom in failed");
  click(scenarios.find(button => button.dataset.s === "t5120"));
  if (api.state().scale !== 1) throw new Error("scenario did not reset view");
});
check("no NaN or undefined render", () => {
  const bad = renders.find(value => /\bNaN\b|\bundefined\b/.test(value));
  if (bad) throw new Error(bad.slice(0, 120));
});
check("embedded JSON script-safe", () => {
  if (payload.includes("<")) throw new Error("unescaped less-than in payload");
});

if (failures.length) {
  console.log("\n" + failures.length + " FAILURES");
  failures.forEach(failure => console.log("- " + failure));
  process.exit(1);
}
console.log("\nALL CHECKS PASSED");
