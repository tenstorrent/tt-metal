# Headless validation harness

There is no browser in this environment, so validate the report's JS by evaluating the inline `<script>`
against a stubbed DOM in node. Node may be at `/proj_sw/user_dev/bsheikh/nodejs/bin/node`.

## What to assert
1. `node --check` on the extracted inline script (syntax) and `JSON.parse` on the `<script type="application/json">` payload.
2. No throw when firing **every** control: each scenario button, each mode button, the Semantic/Ops view
   toggle, a per-node expand, and `openDrawer(<blockId>)`.
3. Structural: semantic view produces `.node` groups + `.edge`s and 0 `.onode`; per-node expand and Ops
   view produce `.onode` (internal op) groups and `.obox`/`.oedge`; the drawer body contains the
   constituent-ops table + tensor edges + a `file:line`.
4. No `NaN` / `undefined ms` / `>undefined` anywhere in the rendered `innerHTML`/`textContent`.
5. Pan/zoom: `gView`/`gBase` set after draw; `zoomAt` changes `gView.w`; opening the drawer PRESERVES
   `gView` (same layout) while a scenario change RESETS it to fit.

## Technique
- Expose the script's inner functions to the harness by appending
  `;globalThis.__api={drawGraph,openDrawer,zoomAt,...};` to the extracted JS string before `eval` — inner
  `function` declarations are otherwise unreachable from the harness scope.
- Stub `document` with a factory returning Proxy-ish elements: `getElementById` memoises by id;
  `querySelectorAll` returns fake buttons carrying the right `dataset` (`{s}`,`{m}`,`{v}`,`{z}`) for each
  `#seg*`/`.gzoom` selector; `createElementNS` returns element stubs; `getBoundingClientRect` returns a
  fixed rect so pan/zoom math runs.
- Seed the `payload` element's `textContent` with the extracted JSON before `eval`.
- Make the element stub's `innerHTML` setter also clear `children` so a single post-draw count is accurate
  (the real `svg.innerHTML=''` clears; a naive stub accumulates across redraws).
- Count nodes by walking `children` and matching the `class` attribute.

A harness error like `expanded is not defined` is a scope artifact (script-local `let`), not a report bug —
reach those via the `__api` shim or by firing a control's `onclick`, not directly.
