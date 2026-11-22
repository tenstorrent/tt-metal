// Runs the teletext page's screen script under node with a fake DOM and a fake clock, then prints the page announced
// at each whole second: node teletext_shim.js <page.html> <seconds> [keys...]. A key "@<t>:<key>" is pressed at second t.
const els = {};
function mk(id){
  const e = {
    id, innerHTML: "", textContent: "", className: "", style: {setProperty(){}}, dataset: {}, attrs: {}, children: [],
    setAttribute(k, v){ this.attrs[k] = v; }, getAttribute(k){ return this.attrs[k]; }, addEventListener(){},
    appendChild(c){ return c; }, closest(){ return null; }, getBoundingClientRect(){ return {height: 480}; },
  };
  if (id === "grid") Object.defineProperty(e, "innerHTML", {
    get(){ return ""; }, set(v){ e.children = Array.from({length: (v.match(/class="row"/g) || []).length}, (_, r) => mk("row" + r)); }});
  return e;
}
const byId = id => els[id] || (els[id] = mk(id));
let now = 0, seq = 0; const timers = new Map(), keyHandlers = [];
globalThis.setInterval = (f, ms) => { const id = ++seq; timers.set(id, {f, ms, at: now + ms}); return id; };
globalThis.clearInterval = id => { timers.delete(id); };
globalThis.setTimeout = (f, ms) => { const id = ++seq; timers.set(id, {f, ms, at: now + ms, once: true}); return id; };
globalThis.clearTimeout = globalThis.clearInterval;
function advance(to){
  for (;;){
    let next = null; for (const [id, t] of timers) if (t.at <= to && (!next || t.at < next[1].at)) next = [id, t];
    if (!next) break; const [id, t] = next; now = t.at;
    if (t.once) timers.delete(id); else t.at += t.ms; t.f();
  }
  now = to;
}
globalThis.window = {matchMedia: () => ({matches: false}), addEventListener(){}};
globalThis.document = {
  getElementById: byId, head: {appendChild(){}}, body: {appendChild(){}}, title: "",
  createElement: () => mk("style"), addEventListener: (ev, f) => { if (ev === "keydown") keyHandlers.push(f); },
};
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
new Function(src.slice(src.indexOf("<script>") + 8, src.lastIndexOf("</script>")))();
const secs = +process.argv[3], keys = process.argv.slice(4).map(k => { const m = /^@(\d+):(.+)$/.exec(k); return {t: +m[1], key: m[2]}; });
const out = [];
for (let s = 1; s <= secs; s++){
  advance(s * 1000);
  for (const k of keys.filter(k => k.t === s)) keyHandlers.forEach(f => f({key: k.key, target: null, preventDefault(){}}));
  out.push(byId("live").textContent.replace(/, subpage.*/, ""));
}
console.log(JSON.stringify(out));
