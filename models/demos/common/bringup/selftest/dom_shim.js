// Minimal DOM stand-in to execute the dashboard script under node: every element accepts the properties and
// calls the script uses, and records innerHTML so the test can check that each section rendered something.
const els = {};
function el(sel){
  if (!els[sel]) els[sel] = {
    sel, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, attrs: {}, outerHTML: "",
    setAttribute(k, v){ this.attrs[k] = v; }, getAttribute(k){ return this.attrs[k]; },
    appendChild(c){ this.innerHTML += "<child>"; return c; }, addEventListener(){},
    querySelector(s){ return el(sel + " " + s); }, querySelectorAll(){ return []; },
    getBoundingClientRect(){ return {left: 0, top: 0, right: 10, width: 900, height: 300}; },
    set onclick(f){}, set onkeydown(f){},
  };
  return els[sel];
}
globalThis.document = {
  querySelector: el, documentElement: {}, title: "",
  createElementNS: (ns, tag) => ({tag, attrs: {}, children: [], textContent: "", setAttribute(k, v){ this.attrs[k] = v; },
    appendChild(c){ this.children.push(c); return c; }}),
};
globalThis.getComputedStyle = () => ({getPropertyValue: () => "#123456"});
globalThis.matchMedia = () => ({addEventListener(){}});
globalThis.MutationObserver = class { observe(){} };
globalThis.innerWidth = 1200; globalThis.innerHeight = 900;
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const script = src.slice(src.indexOf("<script>") + 8, src.lastIndexOf("</script>"));
new Function(script)();
const out = {};
for (const [k, v] of Object.entries(els)) out[k] = {html: v.innerHTML.length, text: v.textContent, hidden: v.hidden};
out.__title = document.title;
console.log(JSON.stringify(out));
